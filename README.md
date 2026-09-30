# SA16 Trot-Bound Switch

Forced Trot/Bound seamless-transition locomotion for the SA16 robot. The PPO
policy outputs a 16D latent action. A forced gait scheduler selects one of two
frozen motion-prior decoders, and the decoded 16D joint action is applied as
`q_des = q_default + 0.25 * decoded_action` with `Kp=40`, `Kd=1`.

Tested in the documented IsaacLab/Isaac Sim environment.

## Environment

- IsaacLab Git commit: `cef06be7ad953a468f2fe4b6e08f1bfde34650fa`
- IsaacLab repository version: 2.3.2
- Isaac Sim: 5.1.0.0
- Python: 3.11.15
- rsl-rl-lib: 5.0.1
- PyTorch: 2.7.0+cu128

The repository uses an existing IsaacLab installation. Set `ISAACLAB_ROOT` if
IsaacLab is not in a sibling directory named `Isaaclab`.

Setup installs this package in editable mode with `--no-deps` and
`--no-build-isolation`; it does not download another IsaacLab installation.

## Setup and playback

```bash
./scripts/setup.sh
./scripts/play_pretrained.sh
```

`play_pretrained.sh` loads the repository-owned `model_1300.pt`, Trot decoder,
Bound decoder, URDF, and meshes. Runtime provenance prints the resolved paths.

## Fresh transition PPO training

```bash
./scripts/train_from_scratch.sh
```

This initializes only the transition PPO actor and critic from scratch. The
Trot and Bound action decoders remain frozen. The formal configuration uses:

- 4096 environments and 1351 iterations
- 62D policy observation and 16D latent action
- approximately 50/50 initial Trot/Bound assignment
- independent forced-gait dwell times sampled uniformly from 3 to 6 seconds
- latent clamp `[-3, 3]`
- transition window of 0.6 seconds
- velocity recovery at XY error below 0.25 m/s for 10 consecutive control steps

The reward weights are `+1.5` linear velocity tracking, `+0.5` angular
velocity tracking, `-0.01` latent smoothness, `-1e-7` joint torque, `-3e-7`
joint acceleration, `-1.5` termination, `-0.5` transition pitch, and `-0.02`
transition pitch rate.

## Evaluation

```bash
./scripts/evaluate.sh src/trot_bound_switch/pretrained/policy/model_1300.pt
```

Evaluation writes its JSON report under `outputs/evaluation/`. Add `--video`
to record a rollout with transition telemetry.

## Integrity and provenance

Verify all packaged binary artifacts with:

```bash
sha256sum --check checksums.sha256
./scripts/smoke_test.sh
```

The package owns the SA16 task, robot config, URDF/meshes, decoders, and policy.
The runtime scripts print every resolved model and asset path for provenance.
The implementation was tested in the documented IsaacLab/Isaac Sim environment;
it has not been claimed as a clean vanilla IsaacLab checkout validation.
