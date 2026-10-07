# Full Training Pipeline

This document describes the code path from random joint-action experts to the
forced Trot/Bound transition policy. The repository ships a known-good final
policy and frozen decoders for immediate playback, but the `artifacts/`
directory produced by this pipeline is intentionally not committed.

## What "from scratch" means

The full pipeline starts both 16D joint-action expert PPO policies from random
actor/critic initialization. It then regenerates the expert datasets, trains a
shared state TVAE, trains one frozen action decoder per gait, and finally starts
the 16D latent transition PPO actor/critic from random initialization.

No stage resumes the packaged transition policy or packaged decoders.

## One-command run

```bash
./scripts/setup.sh
./scripts/validate_pipeline.sh
./scripts/train_full_pipeline.sh
```

The command is intentionally long-running. Outputs default to
`artifacts/full_pipeline/`, which is ignored by Git. Override the location with
`ARTIFACT_ROOT=/path/with/enough/space` when needed.

## Stages

1. `scripts/train_bound_expert.sh`
   Trains the straight-running Bound expert with 4096 environments for 1351
   iterations. Expected checkpoint: `experts/bound/model_1350.pt`.

2. `scripts/train_trot_expert.sh`
   Trains the final turning-capable Trot task directly from random
   initialization with 4096 environments for 2101 iterations. Expected
   checkpoint: `experts/trot/model_2100.pt`.

3. `scripts/collect_expert_datasets.sh`
   Rolls out both experts with 512 environments and saves 2,000,000 valid
   frames per gait. Each frame is an 85D representation state; legal TVAE
   samples are three consecutive frames from one episode.

4. `scripts/train_representation.sh`
   Audits both datasets, creates episode-level train/validation splits,
   computes shared train-only normalization, trains the shared 16D state TVAE,
   and trains the gait-specific 16D-to-16D action decoders.

5. `scripts/train_transition_from_pipeline.sh`
   Loads the newly trained frozen decoders and trains the forced transition PPO
   policy with 4096 environments for 1351 iterations.

The exact numeric recipe is also recorded in `configs/full_pipeline.yaml`.

## Playback and evaluation

After the pipeline finishes:

```bash
./scripts/play_pipeline.sh
./scripts/evaluate_pipeline.sh
```

Both scripts resolve the policy and two decoders from the generated artifact
tree. An explicit transition checkpoint may be supplied as the first argument.

## Historical Trot note

The original successful Trot checkpoint used to collect the reference dataset
was the result of a staged run whose first-stage checkpoint directory was later
deleted. The surviving final task configuration and final checkpoint are known,
but the deleted stage cannot be reconstructed exactly without guessing.

For that reason, the public code-only recipe uses the final turning-capable
Trot configuration directly from random initialization for 2101 iterations.
This is a complete executable training recipe, but it is not claimed to
reproduce the unavailable historical optimizer trajectory step for step.

## Validation scope

`scripts/validate_pipeline.sh` performs static syntax, path, and dependency-file
checks only. Packaging this code did not rerun the multi-day training pipeline.
The original pretrained playback path remains independently available through
`scripts/play_pretrained.sh`.
