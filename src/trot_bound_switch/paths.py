"""Package-owned runtime resources."""

from importlib.resources import files
from pathlib import Path


def resource_path(*parts: str) -> Path:
    """Resolve a package resource to a persistent filesystem path."""
    path = Path(str(files("trot_bound_switch").joinpath(*parts))).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


PACKAGE_ROOT = resource_path()
URDF_PATH = resource_path(
    "assets", "urdf", "装配体_组合_1.SLDASM", "urdf", "装配体23.0_V6.urdf"
)
TROT_DECODER_PATH = resource_path("pretrained", "decoders", "trot_best.pt")
BOUND_DECODER_PATH = resource_path("pretrained", "decoders", "bound_best.pt")
PRETRAINED_POLICY_PATH = resource_path("pretrained", "policy", "model_1300.pt")


def assert_package_owned(path: Path) -> None:
    path.resolve().relative_to(PACKAGE_ROOT)
