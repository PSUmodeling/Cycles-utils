from .control_file import generate_control_file
from .control_file import read_control_file
from .control_file import ControlConfig
from .converter import convert_obsolete_crop_files, convert_obsolete_operation_files, convert_obsolete_soil_files
from .crop_file import read_crop_file, generate_crop_file
from .nudge_file import generate_nudge_file
from .operation_file import read_operation_file
from .operation_file import generate_operation_file
from .operation_file import Operation, Planting, Tillage, Harvest, Kill, FixedFertilization, FixedIrrigation, AutoIrrigation
from .output_file import read_output
from .soil_file import SoilLayer
from .soil_file import generate_soil_file
from .soil_file import read_soil_file
from .weather_file import read_weather_file
from .reinit_file import generate_reinit_file

__all__ = [
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
    "plot_yield",
    "plot_operations",
    "plot_map",
    "plot_satellite_map",
    "read_geospatial_file",
]

# Names that require the optional plotting/geospatial dependencies
# (cartopy, geopandas, fiona). Installed via `pip install cycles-utils[plot]`.
# These are imported lazily (PEP 562) so that `import cycles_tools` and
# `import cycles` work without those heavy dependencies for users who only
# need to read/write Cycles input/output files and run simulations.
_LAZY_ATTRS = {
    "plot_yield": ".plot_tools",
    "plot_operations": ".plot_tools",
    "plot_map": ".plot_tools",
    "plot_satellite_map": ".plot_tools",
    "read_geospatial_file": "._geo_file",
}


def __getattr__(name: str):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib
    try:
        module = importlib.import_module(module_name, __name__)
    except ImportError as exc:
        raise ImportError(
            f"'{name}' requires the optional plotting/geospatial dependencies. "
            f"Install them with: pip install cycles-utils[plot]"
        ) from exc

    value = getattr(module, name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))
