"""Runtime provenance checks for package-owned business dependencies."""

import inspect
import sys
from pathlib import Path

from trot_bound_switch import paths


def provenance(checkpoint: str | Path) -> dict[str, Path]:
    from trot_bound_switch.assets import robot
    from trot_bound_switch.tasks import transition_env_cfg

    resolved = {
        "transition task module": Path(inspect.getfile(transition_env_cfg)).resolve(),
        "robot config module": Path(inspect.getfile(robot)).resolve(),
        "URDF": paths.URDF_PATH,
        "Trot decoder": paths.TROT_DECODER_PATH,
        "Bound decoder": paths.BOUND_DECODER_PATH,
        "policy checkpoint": Path(checkpoint).expanduser().resolve(),
    }
    for name, value in resolved.items():
        print(f"[PROVENANCE] {name}: {value}", flush=True)
    return resolved


def assert_package_business_dependencies(checkpoint: str | Path) -> None:
    resolved = provenance(checkpoint)
    repository_root = paths.PACKAGE_ROOT.parents[1]
    for name, value in resolved.items():
        try:
            value.relative_to(repository_root)
        except ValueError as error:
            raise RuntimeError(f"{name} is outside the new repository: {value}") from error
        if not value.exists():
            raise FileNotFoundError(value)

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
