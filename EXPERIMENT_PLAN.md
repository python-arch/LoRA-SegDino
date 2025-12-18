# Symbolic Alignment for Source-Free Segmentation Adaptation (SegDINO + PEFT)

## Problem Definition
We study **source-free, unlabeled target adaptation** for polyp segmentation (Kvasir-SEG) with a SegDINO-style model (DINOv3 ViT backbone + DPT head).

- **Source training:** uses labeled source-domain data (here: clean Kvasir-SEG `train/`).
- **Target adaptation:** uses **only unlabeled target images** (no source images, no source labels, no source statistics).
- **Evaluation:** uses a **held-out labeled target set** never used for adaptation/tuning.

### Core capability claim
Enable **stable** source-free adaptation under **severe appearance shift** where standard pseudo-label self-training is unreliable, by aligning **learned mask-structure descriptors** during PEFT adaptation (LoRA/SALT and other adapters).

## Experimental Protocol (Reviewer-Proof)
### Data contract (standardized)
Dataset layout (canonical):
- `segdata/kvasir/train/images/*`, `segdata/kvasir/train/masks/*`
- `segdata/kvasir/test/images/*`, `segdata/kvasir/test/masks/*` (provided split)

We further split the provided `test/` once (fixed seed; filelists committed):
- `target_adapt/` (unlabeled, used for adaptation and development iteration)
- `target_holdout/` (labeled, **evaluation only**, never used for adaptation or model selection)

Default: `target_holdout = 20%` of original `test/` (min 15%, max 25%).

### Target shift construction (severity knob)
We construct target domains by applying corruptions to `target_adapt/images` and `target_holdout/images` with **deterministic per-image seeds** (seed = hash(filename, corruption_id, severity)).

#### Stage 1 (main paper): single-family ladders
We run separate severity ladders `S0..S4` for each corruption family:
- Blur
- Noise
- JPEG/compression
- Illumination (brightness/contrast/gamma)

Headline reporting aggregates across families (average AUSC and S4; also provide per-family breakdown).

#### Stage 2 (appendix): mixed corruptions
Per image, compose 1–2 corruptions sampled deterministically (same seed scheme), to emulate “in-the-wild” shifts.

### “Source-free” enforcement
During adaptation:
- No access to `segdata/kvasir/train/*`
- No access to `target_holdout/images`
- No labels used for any selection

## Methods
### Model family
SegDINO-like model: DINOv3 backbone + DPT head (binary segmentation).

Primary adaptation scope (to keep early experiments tight):
- Apply PEFT to the **segmentation head first** (and only expand scope to backbone projections after this is stable).

### Baseline groups (non-negotiable)
1. **No adaptation**
   - Source-only (train on clean, evaluate on target severities)
2. **Source-free adaptation (non-PEFT)**
   - Entropy minimization
   - Augmentation consistency
   - **TENT-style** adaptation (update norm affine params, entropy objective)
3. **Pseudo-label baseline**
   - Self-training with confidence threshold + augmentation (included to demonstrate failure under severe shift)
4. **PEFT-only (non-symbolic)**, adapter ∈ {LoRA, SALT, …}
   - Same best non-symbolic objective as above; update only adapter params
5. **Ours: PEFT + symbolic alignment**, adapter ∈ {LoRA, SALT, …}
   - Learned-symbolic alignment loss + anti-collapse safeguards + same adaptation budget

### PEFT is a pluggable axis
Primary comparisons:
- LoRA vs SALT under **matched trainable parameter budgets**
Secondary (optional): other PEFTs (e.g., IA³/adapters/BitFit) as long as parameter budgets are matched.

## Symbolic Alignment (what “symbolic” means)
We replace hand-crafted shape/topology statistics with a **learned mask-structure descriptor encoder** `E_θ`.

### Learnable symbolic encoder `E_θ` (trained once; then frozen)
`E_θ` maps a mask view to a `k`-dim descriptor:
- Input views:
  - Soft mask `p` (probabilities/logits passed through sigmoid)
  - Boundary view `b(p)` (e.g., boundary band / mask-gradient magnitude / thin edge map)
- Outputs:
  - Global descriptor: `s_g = E_θ(p)`
  - Boundary descriptor: `s_b = E_θ(b(p))` (same encoder, different input view)

Training of `E_θ` happens **once before target adaptation** using **source-domain ground truth masks** from `train/`:
- Use structure-preserving transforms (flip/rotate, resize/crop with aligned remap, conservative morphological perturbations).
- Objective: contrastive (positives = same mask under different transforms; negatives = different masks in batch) or BYOL-style.
- Freeze `E_θ` during target adaptation (first paper version) to reduce instability and avoid leakage concerns.

### Symbolic prior (source-free; target-only)
Primary: **self-bootstrapped EMA prior** on target descriptors:
- Maintain EMA summaries over confident, non-degenerate predictions on `target_adapt/`, separately for:
  - Global: `(μ_g, Σ_g)` (mean + variance/cov)
  - Boundary: `(μ_b, Σ_b)`

Optional: task-knowledge priors (soft ranges) as secondary experiments, if defensible.

### Alignment loss (two-scale)
For each target prediction `p_t`, compute `(s_g(p_t), s_b(p_t))` and align to EMA priors via a robust distance:
- Huber on z-scored deviations, or clipped Mahalanobis distance.

`L_sym = D(s_g, μ_g, Σ_g) + D(s_b, μ_b, Σ_b)`

### Core adaptation objective (non-symbolic)
Primary choice (locked): **augmentation consistency** on unlabeled target images (entropy minimization remains a baseline).

### Anti-collapse safeguards (explicit, separable from symbols)
Non-negotiable safeguards:
- Confidence gating (pixel-/image-level) + warmup/ramp schedule
- Robust loss on symbolic deviations (outlier clipping / Huber / quantile-style)
- Explicit penalties/guards for degenerate masks (all-foreground / all-background)

## Causality Ablations (separate “symbols” vs “safeguards”)
To avoid “it’s just regularization” critiques, include:
- **Safeguards-only** (no symbolic terms)
- **Symbols-only** (no safeguards; expected to collapse; report failure rates)
- **Symbols + safeguards** (full)
- Global-only (remove boundary descriptor)
- Boundary-only (remove global descriptor)

Run ablations at least on `S2` and `S4`, plus include in AUSC if feasible.

## Parameter/Compute Budgets (claims)
Report trainable parameters and time.

Default budgets:
- Main: **0.5% trainable params**
- Curves: **0.1%**, **1%**

All PEFT comparisons must be budget-matched.

## Metrics and Reporting
Primary:
- Dice, IoU

Boundary:
- **Boundary F-score** (primary for endoscopy-style segmentation)
- **HD95** (secondary; framed as robust boundary error under shift)

Stability / failure:
- Empty prediction rate, full prediction rate
- Fragmentation proxy (components/fragmentation)

Headline reporting:
- **AUSC** over `S1–S4`
- Stress-test: **S4** results
- Keep `S0` as sanity check (not the headline)

Reproducibility:
- 3 seeds minimum for headline tables
- Fixed filelists for splits and fixed corruption spec (IDs + severity mapping committed)

## Model selection / stopping (no label leakage)
If any early stopping is used, it must be **unsupervised**, e.g.:
- plateau in target entropy/consistency loss
- stability of symbolic-stat EMA deltas
- “no-collapse” constraints satisfied for N steps

## Minimal Execution Plan (run order)
Phase 1: establish stress regime
1. Train source on clean; evaluate on clean and `S0..S4` per family
2. Identify pseudo-label failure severities (typically S3/S4)

Phase 2: learn the symbolic encoder
3. Train `E_θ` on source masks with structure-preserving transforms; freeze for adaptation
4. Sanity-check `E_θ` invariance qualitatively (same-mask views cluster; different masks separate)

Phase 3: baselines (two severities first)
5. Run non-PEFT baselines on `S2` and `S4`
6. Run PEFT-only LoRA/SALT on `S2` and `S4` (0.5% budget)

Phase 4: ours MVP (S4-first)
7. Ours (learned symbols + safeguards) on `S4` first, then sweep `S1..S4`

Phase 5: ablations + budget curves
8. Causality ablations on `S2` and `S4`
9. Budget curves at `S4` + AUSC for LoRA vs SALT (0.1/0.5/1%)

Appendix: mixed corruptions; optional “domain dial” (scale adapter strength at inference).

## Figures Checklist (paper narrative)
1. Protocol schematic: source train → target_adapt → target_holdout (+ severities)
2. Source-only degradation: Dice/IoU vs severity (per family)
3. Descriptor learning sanity: UMAP/t-SNE of `E_θ` outputs (optional but strong)
4. Main results: AUSC and S4 (methods grouped: non-PEFT, PEFT-only, ours)
4. Pseudo-label failure: self-training collapses at S4; ours remains stable (include failure rates)
5. Causality ablations: safeguards-only vs symbols-only vs full (+ term removals)
6. Efficiency curve: performance vs trainable params (LoRA vs SALT)
7. Qualitative grids on S4 (input/pred/gt) + symbolic stats over adaptation steps
