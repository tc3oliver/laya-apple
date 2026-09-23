"""Apple-native inference runtime for Laya: MLX GPU by default, validated fixed-shape ANE paths."""

__version__ = "1.0.2"

from .errors import (  # noqa: E402
    ArtifactError,
    ArtifactIntegrityError,
    ArtifactMissingError,
    ArtifactParityError,
    ArtifactRevisionError,
    BackendUnavailableError,
    ComputeUnitMismatchError,
    InvalidRequestError,
    LayaAppleError,
    UnsupportedModelError,
    UnsupportedShapeError,
)
from .model import Laya  # noqa: E402
from .result import Result, RuntimeInfo  # noqa: E402

__all__ = [
    "Laya",
    "Result",
    "RuntimeInfo",
    "LayaAppleError",
    "UnsupportedModelError",
    "InvalidRequestError",
    "UnsupportedShapeError",
    "BackendUnavailableError",
    "ArtifactError",
    "ArtifactMissingError",
    "ArtifactRevisionError",
    "ArtifactIntegrityError",
    "ArtifactParityError",
    "ComputeUnitMismatchError",
    "__version__",
]
