from .cycles import Cycles
from .cycles_tools import generate_control_file
from .cycles_tools import generate_nudge_file
from .cycles_tools import generate_soil_file
from .cycles_tools import generate_reinit_file
from .cycles_tools import read_control_file
from .cycles_tools import read_crop_file
from .cycles_tools import generate_crop_file
from .cycles_tools import read_soil_file
from .cycles_tools import read_weather_file
from .cycles_tools import read_output
from .cycles_tools import read_operation_file
from .cycles_tools import generate_operation_file
from .cycles_tools import convert_obsolete_crop_files
from .cycles_tools import convert_obsolete_operation_files
from .cycles_tools import convert_obsolete_soil_files
from .cycles_tools import SoilLayer
from .cycles_tools import Operation, Planting, Tillage, Harvest, Kill, FixedFertilization, FixedIrrigation, AutoIrrigation
from .cycles_runner import CyclesRunner
from .rotation_builder import CyclesRotationBuilder, Crop, CropGroup

__all__ = [
    "Cycles",
    "CyclesRunner",
    "CyclesRotationBuilder",
    "Crop",
    "CropGroup",
    "generate_control_file",
    "read_control_file",
    "generate_nudge_file",
    "read_crop_file",
    "generate_crop_file",
    "read_operation_file",
    "generate_operation_file",
    "read_output",
    "generate_soil_file",
    "read_soil_file",
    "read_weather_file",
    "generate_reinit_file",
    "convert_obsolete_crop_files",
    "convert_obsolete_operation_files",
    "convert_obsolete_soil_files",
    "SoilLayer",
    "Operation", "Planting", "Tillage", "Harvest", "Kill",
    "FixedFertilization", "FixedIrrigation", "AutoIrrigation",
    "plot_yield",
    "plot_operations",
    "plot_map",
    "plot_satellite_map",
]

# Plotting helpers require the optional [plot] extra (cartopy, geopandas,
# fiona). Import lazily so `import cycles` never requires them.
_PLOT_ATTRS = ("plot_yield", "plot_operations", "plot_map", "plot_satellite_map")


def __getattr__(name: str):
    if name in _PLOT_ATTRS:
        from . import cycles_tools
        value = getattr(cycles_tools, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_PLOT_ATTRS))
