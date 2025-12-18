# Implementation Plan (Staged) — Symbolic Alignment + Learned Descriptors

This document breaks the experimental protocol in `EXPERIMENT_PLAN.md` into concrete implementation stages, with intended artifacts, interfaces, and validation checks. It is an execution plan only (no code in this step).

## Guiding principles
- **Repro first:** deterministic splits, deterministic corruptions, committed manifests.
- **Source-free strictness:** adaptation code must never read `train/` and must never see `target_holdout/`.
- **Pluggable axes:** PEFT adapter type and loss components (core/symbolic/safeguards) are modular and switchable.
- **Minimal intrusion:** keep existing scripts working where possible; add new modules gradually.

## Proposed repo structure (incremental)
We keep top-level entry scripts, but factor reusable components into a small package:
- `segdino/` (new): datasets, corruptions, adapters, symbolic encoder, losses, metrics, runners
- `scripts/` (optional later): thin wrappers / CLI entrypoints
- `segdata/` (external, not committed): datasets and generated corruptions
- `splits/` (new, committed): filelists and split metadata
- `configs/` (new, committed): experiment configs and sweep templates

## Current repo mapping → staged refactor (keep scripts working)
This repo currently uses top-level scripts and utilities:
- `dataset.py`: folder dataset + resize/normalize
- `dpt.py` / `blocks.py`: segmentation head
- `train_segdino.py`: baseline training
- `PEFT_segdino.py`: PEFT training + (currently) wandb logging
- `test_segdino.py`: evaluation metrics + visualization

### Migration strategy (no breakage)
We will not “big bang” rewrite. Instead:
1. Introduce a new internal package (`segdino/`) containing reusable components.
2. Keep existing entrypoints working by:
   - leaving their CLI intact where possible
   - gradually replacing internal logic with imports from `segdino/`
3. Add **new** entrypoints only when needed (e.g., `train_symbolic_encoder.py`, `adapt.py`) so the baseline scripts remain usable.

### What gets moved/duplicated first vs later
Early stages (unblock experiments):
- `dataset.py` functionality becomes `segdino/data.py` (manifest selection + target splits + view generation), but we can keep `dataset.py` as a thin wrapper initially.
- Corruptions and deterministic seeding are new (`segdino/corruptions.py`) and used by new adaptation/eval paths first.
- Metrics from `test_segdino.py` (Dice/IoU/HD95) become `segdino/metrics.py`; `test_segdino.py` can import them.

Later stages (after results are flowing):
- Consolidate duplicated training loops between `train_segdino.py` and `PEFT_segdino.py` into a shared runner module (optional).
- Normalize logging (CSV + optional wandb) behind a small interface to avoid coupling experiments to wandb.

### “Known sharp edges” to address during migration
- **`dinov3/` dependency:** scripts expect a local torch.hub repo under `--repo_dir` (needs `hubconf.py`). We will document and validate this early to avoid silent failures.
- **Dataset directory naming:** standardize to `images/` + `masks/` everywhere; keep backwards-compatible CLI flags for older names if needed.
- **W&B coupling:** `PEFT_segdino.py` currently initializes W&B unconditionally; we will gate it so local runs don’t fail when wandb is absent (when we reach that file).

## Stage 0 — Baseline wiring & repo hygiene
**Goal:** ensure there is a single canonical dataset naming convention and minimal config drift.

Deliverables:
- Confirm `images/` + `masks/` everywhere (CLI defaults; docs).
- A single “paths and splits” contract referenced by all training/adaptation/eval entrypoints.
- Optional: a short `README` section pointing to `EXPERIMENT_PLAN.md`.

Validation:
- Dry-run CLI parsing and path resolution without reading data.

## Stage 1 — Deterministic split manifests (target_adapt vs target_holdout)
**Goal:** create `target_adapt/` and `target_holdout/` logically (and optionally physically) with committed filelists.

Tasks:
- Implement a splitter that:
  - reads `segdata/kvasir/test/images` filenames
  - produces `splits/kvasir_target_adapt.txt` and `splits/kvasir_target_holdout.txt`
  - uses fixed seed and stable sorting
- Decide whether to:
  - (A) keep images in place and drive selection via manifests only (preferred), or
  - (B) materialize `target_adapt/` and `target_holdout/` folders (higher I/O).

Deliverables:
- `splits/` text manifests + a `splits/metadata.json` capturing seed, holdout %, creation time.

Validation:
- Consistency check: no overlaps; counts match expected ratios.
- Dataset loader can load “subset by manifest” and return paired image/mask paths.

## Stage 2 — Deterministic corruption pipeline (single-family ladders + mixed)
**Goal:** create a reproducible corruption system with severity `S0..S4`, per-family ladders first.

Tasks:
- Define a corruption spec:
  - families: blur, noise, jpeg, illumination
  - severity mapping `S0..S4` per family
  - deterministic per-image RNG seed = f(filename, family, severity)
- Decide caching strategy:
  - (A) on-the-fly corruptions in the dataset pipeline (fast iteration; deterministic), or
  - (B) offline materialization into `segdata/kvasir_<family>_S{0..4}/...` (faster training; more disk)
  - Recommended: start on-the-fly; add offline cache option later.
- Implement “mixed corruption” mode for appendix:
  - choose 1–2 corruptions per image deterministically and compose.

Deliverables:
- `segdino/corruptions.py` spec (family implementations + severity maps + seeding rules).
- Config representation: `configs/corruptions/*.yaml` (or JSON) with the fixed IDs and mappings.

Validation:
- Determinism test: same image+spec yields identical pixels across runs/machines.
- Mask alignment check: masks remain unchanged and correctly paired.

## Stage 3 — Unified dataset + augmentation views for consistency training
**Goal:** one dataset interface that supports:
- clean source training
- corrupted target adaptation
- paired weak/strong views for consistency objectives
- subset selection by manifest

Tasks:
- Standardize sample dict fields: `{image, mask(optional), id, path, domain_meta}`
- Add a view generator:
  - `weak_view(x)` and `strong_view(x)` for consistency objective
  - ensure augmentations are deterministic under a seed when needed for debugging

Deliverables:
- `segdino/data.py` (dataset + transforms + manifest selection).
- One CLI surface shared by train/adapt/test scripts for `--dataset_root`, `--split_manifest`, `--img_dir_name images`, `--label_dir_name masks`.

Validation:
- Smoke-load 10 samples from each split and each severity.

## Stage 4 — Learned symbolic descriptor encoder `E_θ` (pretrain once)
**Goal:** implement and train a tiny mask-structure encoder on **source masks only**, then freeze it.

Design constraints (from protocol):
- Input views: soft mask-like tensors and boundary-band tensors
- Tiny architecture (2–4 conv blocks + MLP), `k=32/64`
- Training: contrastive or BYOL-style on structure-preserving mask augmentations

Tasks:
- Implement mask augmentation pipeline for descriptor learning (structure-preserving).
- Implement `E_θ` model + projection head (as needed).
- Implement training script:
  - trains on `segdata/kvasir/train/masks`
  - saves `E_θ` checkpoint + config

Deliverables:
- `segdino/symbolic_encoder.py` (model + boundary view function).
- `train_symbolic_encoder.py` (entrypoint).
- Saved weights path convention under `runs/` (ignored by git).

Validation:
- Qualitative: retrieval sanity (nearest neighbors in embedding space).
- Optional: UMAP/t-SNE figure generator for paper.

## Stage 5 — Pluggable PEFT adapter framework (budget-matched)
**Goal:** make adapter type a config switch: LoRA, SALT, and future PEFTs.

Tasks:
- Define a minimal adapter API:
  - `apply_adapter(model, spec) -> adapted_model`
  - `count_trainable_params(model)` utility
- Implement adapters:
  - LoRA on selected linear layers
  - SALT on selected linear layers
  - (optional later) IA³ / adapters / BitFit
- Implement “budget matching” helpers:
  - given a target % budget, select ranks or layer subsets to match within tolerance

Deliverables:
- `segdino/adapters/` package with `lora.py`, `salt.py`, `registry.py`.
- A config schema for adapter specs (`configs/adapters/*.yaml`).

Validation:
- Unit-ish checks (fast):
  - adapted model forward matches shape
  - only adapter params require grad
  - trainable param counts match requested budget roughly

## Stage 6 — Core adaptation objectives + baselines (source-free)
**Goal:** implement baselines and the core objective used by your method.

Primary core objective (locked):
- augmentation consistency (weak/strong views), plus optional entropy term

Baselines to implement:
- source-only evaluation pipeline
- entropy minimization
- consistency-only
- TENT-style (norm affine only)
- pseudo-label self-training (with confidence threshold)

Deliverables:
- `segdino/objectives.py` with modular losses
- `adapt.py` entrypoint supporting method presets via flags/config

Validation:
- Overfit check on a tiny subset (sanity) and confirm losses decrease.
- Confirm adaptation runner never touches source split paths (explicit path guard).

## Stage 7 — Symbolic EMA priors + two-scale alignment loss (global + boundary)
**Goal:** implement the key contribution: learned symbolic alignment with strict gating and robust distances.

Tasks:
- Compute descriptors during adaptation:
  - `s_g = E_θ(p)`
  - `s_b = E_θ(boundary_view(p))`
- Maintain EMA stats on `target_adapt` confident predictions:
  - mean + (diag variance initially; full cov optional later)
- Implement gating:
  - image-level confidence
  - degenerate mask checks
  - fragmentation proxy bounds
- Implement robust distance:
  - z-score + Huber, or clipped Mahalanobis
- Implement warmup/ramp schedule for `λ_sym`

Deliverables:
- `segdino/symbolic_prior.py` (EMA stats + update rules + gating)
- `segdino/symbolic_loss.py` (two-scale alignment)

Validation:
- Logging: distribution of accepted/rejected samples; EMA drift curves.
- Failure-rate reduction vs PEFT-only on S4 (quick regression test).

## Stage 8 — Metrics: Dice/IoU + Boundary F-score + HD95 + failure rates
**Goal:** compute the locked metrics consistently across all runs.

Tasks:
- Implement:
  - Dice, IoU
  - Boundary F-score (with fixed tolerance in pixels; document it)
  - HD95 (existing in `test_segdino.py` can be reused/ported)
  - failure rates and fragmentation proxy

Deliverables:
- `segdino/metrics.py` used by evaluation scripts
- CSV outputs compatible with aggregate scripts

Validation:
- Metric sanity on a few hand-checked examples (empty/full masks).

## Stage 9 — Experiment orchestration (configs, sweeps, multi-node)
**Goal:** make the run matrix easy to launch and hard to mess up.

Tasks:
- Add config templates:
  - corruption family + severity
  - adapter type + budget
  - method preset (baseline vs ours vs ablation)
- Add a launcher script that expands a grid into commands (or integrate with W&B sweeps if desired).
- Ensure every run writes:
  - config snapshot
  - seed
  - split manifests
  - corruption spec ID

Deliverables:
- `configs/` for reproducible runs
- `scripts/launch_grid.py` (optional) or W&B sweep YAMLs updated to new CLI

Validation:
- “Resume and rerun” determinism: rerunning same config reproduces identical corruption and split selection.

## Stage 10 — Paper-facing artifacts
**Goal:** produce the figures/checklists specified in `EXPERIMENT_PLAN.md`.

Tasks:
- Add plotting/aggregation utilities:
  - AUSC computation over severities
  - per-family and averaged tables
  - plots: degradation curves, budget curves, failure rates
  - optional: UMAP/t-SNE for `E_θ`

Deliverables:
- `analysis/` or `notebooks/` (optional) for aggregation scripts
- One “make figures” doc with exact commands

Validation:
- End-to-end reproduction on a new machine using only configs + manifests.

## Implementation order (what we will actually do next)
1. Stage 1 (split manifests) + Stage 3 (unified dataset selection) — unblock everything else.
2. Stage 2 (corruptions) — produce severity ladders and source-only degradation curves.
3. Stage 5 (pluggable adapters) + Stage 6 (core objectives + baselines).
4. Stage 4 (`E_θ` training) — train once and freeze.
5. Stage 7 (symbolic EMA + alignment) — implement full method + causality ablations.
6. Stage 8–10 (metrics, orchestration, paper artifacts).
