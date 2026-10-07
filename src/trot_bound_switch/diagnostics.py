"""Runtime provenance checks for package-owned business dependencies."""

import inspect
import sys
from pathlib import Path

from trot_bound_switch import paths


def provenance(
    checkpoint: str | Path,
    trot_decoder: str | Path | None = None,
    bound_decoder: str | Path | None = None,
) -> dict[str, Path]:
    from trot_bound_switch.assets import robot
    from trot_bound_switch.tasks import transition_env_cfg

    resolved = {
        "transition task module": Path(inspect.getfile(transition_env_cfg)).resolve(),
        "robot config module": Path(inspect.getfile(robot)).resolve(),
        "URDF": paths.URDF_PATH,
        "Trot decoder": Path(trot_decoder).expanduser().resolve() if trot_decoder else paths.TROT_DECODER_PATH,
        "Bound decoder": Path(bound_decoder).expanduser().resolve() if bound_decoder else paths.BOUND_DECODER_PATH,
        "policy checkpoint": Path(checkpoint).expanduser().resolve(),
    }
    for name, value in resolved.items():
        print(f"[PROVENANCE] {name}: {value}", flush=True)
    return resolved


def assert_package_business_dependencies(
    checkpoint: str | Path,
    trot_decoder: str | Path | None = None,
    bound_decoder: str | Path | None = None,
    allow_external_models: bool = False,
) -> None:
    resolved = provenance(checkpoint, trot_decoder=trot_decoder, bound_decoder=bound_decoder)
    repository_root = paths.PACKAGE_ROOT.parents[1]
    for name, value in resolved.items():
        if not value.exists():
            raise FileNotFoundError(value)
        is_model = name in {"Trot decoder", "Bound decoder", "policy checkpoint"}
        if allow_external_models and is_model:
            continue
        try:
            value.relative_to(repository_root)
        except ValueError as error:
            raise RuntimeError(f"{name} is outside the new repository: {value}") from error

    legacy = {
        name: getattr(module, "__file__", None)
        for name, module in sys.modules.items()
        if "config.sa_16dof" in name or name.endswith("robots.sa_16dof")
    }
    print(f"[PROVENANCE] legacy SA16 module count: {len(legacy)}", flush=True)
    if legacy:
        for name, source in sorted(legacy.items()):
            print(f"[PROVENANCE][FAIL] legacy module: {name} -> {source}", flush=True)
        raise RuntimeError("Legacy SA16 modules were imported")
    print("[PROVENANCE] repository ownership checks passed", flush=True)
