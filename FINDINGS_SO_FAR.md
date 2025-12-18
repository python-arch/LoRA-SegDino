# Findings so far (Kvasir + corruption ladder + source-free baselines)

This note records what we observed empirically so far, and why the results make sense. It is meant as a running lab notebook that can later be distilled into the paper narrative.

## Setup recap (what is being tested)
- Task: binary polyp segmentation (Kvasir-SEG).
- Protocol: **source training** on `train/` only; **source-free target adaptation** on `target_adapt` only; evaluation only on `target_holdout`.
- Target shift: deterministic corruptions applied to target images with a severity ladder `S0..S4`.
- Stress regime focus: **mixed corruptions** where pseudo-label self-training is expected to be unreliable.

## Splits are correct (source-free enforcement foundation)
We generated deterministic manifests:
- `splits/kvasir_target_adapt.txt`: 160 images
- `splits/kvasir_target_holdout.txt`: 40 images
- Verified no overlap between the two (0 overlapping lines).

This is critical because it ensures:
- adaptation never sees holdout images (no test leakage),
- the holdout size is enough to estimate Dice/HD95 trends,
- the evaluation protocol is reviewer-proof.

## Source-only degradation curves (why blur/jpg/illumination looked “too easy”)
### Observation
On the initial single-family ladders `S0..S4`:
- **Blur**: Dice stayed ~flat (near invariant).
- **JPEG**: small/no consistent degradation.
- **Illumination**: mild degradation.
- **Noise**: the only family that produced a meaningful drop at `S4`.

### Interpretation / reasoning
This is plausible because:
- DINOv3-style features can be relatively robust to moderate blur/JPEG changes, especially after resizing to a fixed input resolution.
- Blur can sometimes act like denoising and reduce high-frequency clutter, causing non-monotone behavior on small evaluation sets.
- Noise directly corrupts fine texture and local cues; segmentation is more sensitive to this.

Conclusion: **single-family blur/JPEG/illumination at S0..S4 were not “hard” enough** to create the desired pseudo-label failure regime.

## Mixed corruptions created the desired “pseudo-label failure” regime
### Observation
We evaluated mixed corruptions across severities for `num_ops=2` and `num_ops=4`:
- `mixed ops2`:
  - `S4` Dice ≈ **0.8187** (mild shift; small drop from clean).
- `mixed ops4`:
  - `S4` Dice ≈ **0.7188** (large drop).
  - Boundary F-score drops sharply; HD95 increases (worse boundary errors).
  - No degenerate collapse in source-only evaluation (`empty_rate=0`, `full_rate=0`).

### Interpretation / reasoning
This is exactly the “stress knob” we need:
- `ops2` is a “moderate” shift regime where reasonable methods should stay close to source-only.
- `ops4 S4` is a “severe” shift regime where:
  - pseudo-label quality is expected to deteriorate,
  - boundary metrics degrade strongly,
  - adaptation methods have room to improve.

This supports the protocol design: **use `mixed ops4 S4` as the stress-test regime** and `mixed ops2` as a moderate regime.

## First attempt at adaptation baselines collapsed (and why)
### Observation
Initial baseline adaptation runs (entropy / consistency / self-train / tent) collapsed to **all-empty predictions** (empty_rate=1.0).

### Interpretation / reasoning
This is a known failure mode:
- Entropy minimization has a trivial solution for binary segmentation: push probabilities toward 0 everywhere.
- Consistency can also collapse if the model is free to drift without a stabilizer.
- Self-training collapses when pseudo-labels are poor and feedback loops reinforce errors.

This was *useful* diagnostically: it showed our initial baseline implementation was missing standard stabilizers needed for severe shifts.

## Stabilized source-free baselines (teacher regularization + restricted trainable scope)
### Implementation changes that fixed collapse
We added two key stabilizers:
1) **Trainable scope restriction (default head-only)** for non-TENT baselines, to avoid catastrophic drift.
2) A **frozen teacher copy** of the initial model and a KL regularizer `KL(teacher || student)` on weak-view predictions, plus using teacher predictions for pseudo-label generation in self-training.

### Observation (mixed ops4 S4, 500 steps, bs=4, lr=1e-4)
All methods became stable (no collapse, empty/full rates = 0):
- `entropy`: Dice **0.7258**, IoU **0.6070**, BF **0.2472**, HD95 **48.83**
- `consistency`: Dice **0.7235**, IoU **0.6029**, BF **0.2503**, HD95 **48.59**
- `selftrain`: Dice **0.7145**, IoU **0.5968**, BF **0.2292**, HD95 **44.15**
- `tent`: Dice **0.7349**, IoU **0.6152**, BF **0.2289**, HD95 **46.92**

Relative to source-only `mixed ops4 S4` (Dice ~0.7188):
- entropy / consistency: small gains (≈ +0.5 to +0.7 Dice points)
- selftrain: slightly worse than source-only (consistent with “pseudo-labels are unreliable”)
- tent: best Dice/IoU among non-symbolic baselines here

### Interpretation / reasoning
This is consistent with expectations:
- With a strong frozen teacher, entropy/consistency behave more like “small refinement” rather than unconstrained drift.
- Self-training remains sensitive to pseudo-label noise even with stabilization; under severe shift it can underperform simpler methods.
- TENT often gives modest gains by adjusting normalization statistics/affines without changing the model’s core semantics.

## Practical takeaways for the next stage (learned-symbolic method)
- The regime `mixed ops4 S4` is sufficiently hard to be a meaningful stress test.
- We now have stable baselines to beat; **TENT is currently the strongest** on Dice/IoU in this regime.
- A credible contribution must improve:
  - Dice/IoU **and/or**
  - boundary metrics (BF, HD95) **and**
  - maintain low failure rates (no collapse).

## Known limitations of current evidence
- Holdout is 40 images (fine for iteration but still somewhat noisy); we should run 3 seeds later.
- We haven’t yet produced the per-image pseudo-label confidence/entropy plots that justify “pseudo-labels are poor” in a figure; that’s the next diagnostic.

