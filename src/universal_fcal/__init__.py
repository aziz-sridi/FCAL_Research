"""Architecture-independent FCAL objectives and automatic control.

The package accepts signed class-evidence maps with any spatial resolution.  It
does not know about ResNet, ViT, or a particular dataset; model adapters only
need to provide ``[B, H, W]`` evidence tensors.
"""

from .gradient import GradientStats, gradient_safe_backward
from .objective import (
    FCALOutput,
    MapMeasurements,
    UniversalFCAL,
    budgeted_objective,
    measure_maps,
    resolution_aware_tolerance_pixels,
)
from .scheduler import AutoFCALScheduler

__all__ = [
    "AutoFCALScheduler",
    "FCALOutput",
    "GradientStats",
    "MapMeasurements",
    "UniversalFCAL",
    "budgeted_objective",
    "gradient_safe_backward",
    "measure_maps",
    "resolution_aware_tolerance_pixels",
]
