# SA16 Trot-Bound Switch

SA16 locomotion code and artifacts for the complete path from gait experts to
forced Trot/Bound seamless-transition control.

The final PPO policy outputs a 16D latent action. A forced gait scheduler
selects one of two frozen motion-prior decoders, and the decoded 16D joint
action is applied as `q_des = q_default + 0.25 * decoded_action` with `Kp=40`
and `Kd=1`.

## Environment

- IsaacLab Git commit: `cef06be7ad953a468f2fe4b6e08f1bfde34650fa`
- IsaacLab repository version: 2.3.2
- Isaac Sim: 5.1.0.0
- Python: 3.11.15
- rsl-rl-lib: 5.0.1
- PyTorch: 2.7.0+cu128

Tested in the documented IsaacLab/Isaac Sim environment. This package installs
beside an existing IsaacLab checkout and does not modify IsaacLab source files.
Set `ISAACLAB_ROOT` when IsaacLab is not in a sibling directory named
`Isaaclab`.

## Setup

```bash
git clone https://github.com/fanLnen/trot-bound_switch.git
cd trot-bound_switch
./scripts/setup.sh
./scripts/validate_pipeline.sh
```

Setup uses the existing IsaacLab Python environment with `--no-deps` and
`--no-build-isolation`; it does not download another IsaacLab installation.

## Play the pretrained result

```bash
./scripts/play_pretrained.sh
```

This loads the repository-owned `model_1300.pt`, Trot decoder, Bound decoder,
URDF, and meshes. Runtime provenance prints every resolved path.

## Train the complete pipeline

```bash
./scripts/train_full_pipeline.sh
```

This code-only pipeline performs five stages:

1. Fresh 16D joint-action Bound expert PPO: 4096 environments, 1351 iterations.
2. Fresh 16D joint-action Trot expert PPO: 4096 environments, 2101 iterations.
3. Expert rollout collection: 512 environments and 2,000,000 frames per gait.
4. Shared state TVAE plus separate Trot and Bound action decoders.
5. Fresh forced-transition latent PPO: 4096 environments, 1351 iterations.

Generated checkpoints and datasets are written under
`artifacts/full_pipeline/` and are ignored by Git. The two raw datasets require
roughly 2.2 GB, with additional space required for checkpoints and intermediate
representation files. Set `ARTIFACT_ROOT` to use another location.

Individual stages can also be run with:

```bash
./scripts/train_bound_expert.sh
./scripts/train_trot_expert.sh
./scripts/collect_expert_datasets.sh
./scripts/train_representation.sh
./scripts/train_transition_from_pipeline.sh
```

For backward compatibility, `scripts/train_from_scratch.sh` remains the older
transition-PPO-only entry point and uses the packaged frozen decoders. Use
`train_full_pipeline.sh` for the expert-to-transition workflow.

The complete recipe, stage contracts, expected outputs, and the historical
Trot-training limitation are documented in
[`docs/FULL_PIPELINE.md`](docs/FULL_PIPELINE.md). Exact numeric settings are in
[`configs/full_pipeline.yaml`](configs/full_pipeline.yaml).

Packaging this code did not rerun the multi-day full pipeline. Static checks do
not constitute a claim that every newly trained seed will numerically match the
packaged policy.

## Play or evaluate a newly trained pipeline

```bash
./scripts/play_pipeline.sh
./scripts/evaluate_pipeline.sh
```

Both commands default to the generated `model_1300.pt` and generated Trot/Bound
decoders. An explicit transition checkpoint can be passed as the first
argument.

The packaged reference policy can be evaluated independently:

```bash
./scripts/evaluate.sh src/trot_bound_switch/pretrained/policy/model_1300.pt
```

Add `--video` to either evaluation command to record a rollout with transition
telemetry.

## Transition configuration

- 62D policy observation and 16D latent action
- approximately 50/50 initial Trot/Bound assignment
- independent forced-gait dwell times sampled uniformly from 3 to 6 seconds
- latent clamp `[-3, 3]`
- transition window of 0.6 seconds
- velocity recovery at XY error below 0.25 m/s for 10 consecutive control steps
- no latent-action symmetry augmentation

Reward weights are `+1.5` linear velocity tracking, `+0.5` angular velocity
tracking, `-0.01` latent smoothness, `-1e-7` joint torque, `-3e-7` joint
acceleration, `-1.5` termination, `-0.5` transition pitch, and `-0.02`
transition pitch rate.

## Integrity

```bash
sha256sum --check checksums.sha256
./scripts/smoke_test.sh
```

The package owns the SA16 robot config, URDF/meshes, expert tasks and rewards,
expert symmetry and curriculum, data collector, representation training code,
transition task, frozen reference decoders, and reference final policy. It does
not import the old custom SA16 task modules from a local IsaacLab source tree.
