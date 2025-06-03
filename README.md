# Cycles-utils

`Cycles-utils` is a Python package designed to facilitate [Cycles](https://github.com/PSUmodeling/Cycles) agroecosystem model simulations.
This package provides a number of tools for users to prepare Cycles simulation input files, run Cycles simulations, and post-process Cycles simulation results.

For usage examples, please refer to this [Jupyter notebook](https://github.com/PSUmodeling/Cycles/blob/master/cycles-utils.ipynb).

# Installation

To install:

```shell
pip install Cycles-utils
```

# API reference

## Cycles input/output

### cycles.generate_control_file

`cycles.generate_control_file(filename, user_control_dict)`

Generate a Cycles simulation control file from a user-defined dictionary of control parameters.

**filename**: str

&nbsp;&nbsp;&nbsp;&nbsp; Name (including path) of control file to be generated.

**user_control_dict**: dict of {parameter: value}

&nbsp;&nbsp;&nbsp;&nbsp; Control file parameters and values. The following parameters must be included in the dictionary:
- `simulation_start_date`
- `simulation_end_date`
- `rotation_size`
- `operation_file`
- `soil_file`
- `weather_file`

&nbsp;&nbsp;&nbsp;&nbsp; All other parameters are optional. The following default values are used for optional parameters, but can be overridden by `user_control_dict`:
- `reinit_file`: `'N/A'`
- `co2_level`: `'-999'`
- `use_reinitialization`: `0`
- `adjusted_yields`: `0`
- `automatic_nitrogen`: `0`
- `automatic_phosphorus`: `0`
- `automatic_sulfur`: `0`

&nbsp;&nbsp;&nbsp;&nbsp; All output control parameters are set to `0` by default.
Please refer to the [Cycles User Reference Guide](https://psumodeling.github.io/Cycles/#simulation-control-file-ctrl) for detailed description of control parameters.

### cycles.generate_soil_file

`cycles.generate_soil_file(filename, soil_df, description='', hsg='', slope=0.0, soil_depth=None)`

Generate a Cycles soil description file from a pandas DataFrame containing soil profile parameters.

**filename**: str

&nbsp;&nbsp;&nbsp;&nbsp; Name (including path) of soil file to be generated

**soil_df**: pandas.DataFrame

&nbsp;&nbsp;&nbsp;&nbsp; Pandas DataFrame that contains soil profile parameters, including layer thicknesses, clay contents, sand contents, soil organic carbon, and bulk densities.

**description**: str, optional

&nbsp;&nbsp;&nbsp;&nbsp; Description of soil file that can be added to the soil file as comment.

**hsg**: str, optional

&nbsp;&nbsp;&nbsp;&nbsp; Soil hydrologic groups, designated A, B, C, or D,

**slope**: float, optional

&nbsp;&nbsp;&nbsp;&nbsp; Slope of field in %, or units of rise per 100 units of run.

**soil_depth**: float, optional

&nbsp;&nbsp;&nbsp;&nbsp; Total soil depth. If not provided, the minimum of total soil depth in `soil_df` and 2.0 m will be used.

### cycles.read_output

`cycles.read_output(cycles_path, simulation, output)`

Read Cycles output file.

**cycles_path**: str

&nbsp;&nbsp;&nbsp;&nbsp; Path that contains the Cycles `output` directory.

**simulation**: str

&nbsp;&nbsp;&nbsp;&nbsp; Name of simulation.

**output**: str

&nbsp;&nbsp;&nbsp;&nbsp; Type of output file, e.g., `harvest`.

**Returns:** A pandas DataFrame containing output and a dictionary containing variable units.

### cycles.read_operations

`cycles.read_operations(cycles_path, operation)`

Read Cycles management operations input file.

**cycles_path**: str

&nbsp;&nbsp;&nbsp;&nbsp; Path that contains the Cycles `input` directory.

**operation**: str

&nbsp;&nbsp;&nbsp;&nbsp; Name of operation file.

**Returns:**

A pandas DataFrame containing management operations.

### cycles.read_weather

`cycles.read_weather(cycles_path, weather, start_year=0, end_year=9999)`

Read Cycles weather input file.

**cycles_path**: str

&nbsp;&nbsp;&nbsp;&nbsp; Path that contains the Cycles `input` directory.

**weather**: str

&nbsp;&nbsp;&nbsp;&nbsp; Name of weather file.

**start_year, end_year**: int, optional

&nbsp;&nbsp;&nbsp;&nbsp; Start and end years to be read.

**Returns:** A pandas DataFrame containing daily weather, from `start_year` to `end_year` if applicable.

## Cycles visualization

### cycles.plot_map

`cycles.plot_map(gdf, column, projection=cartopy.crs.PlateCarree(), cmap='viridis', map_axes=None, colorbar_axes=None, title=None, vmin=None, vmax=None, extend='neither', colorbar_orientation='horizontal', fontsize=None)`

Plot variables on a map.

**gdf**: geopandas.GeoDataFrame

&nbsp;&nbsp;&nbsp;&nbsp; A geopandas GeoDataFrame that contains the spatial information for plotting.

**column**: str

&nbsp;&nbsp;&nbsp;&nbsp; Column of the `gdf` for plotting.

**projection**: cartopy.crs.CRS

**cmap**: matplotlib.colors.Colormap

**map_axes, colorbar_axes**: tuple[float, float, float, float], optional

&nbsp;&nbsp;&nbsp;&nbsp; Map and color bar axes in the plot.

**title**: str or None

&nbsp;&nbsp;&nbsp;&nbsp; Plot title.

**vmin, vmax**: float, optional

&nbsp;&nbsp;&nbsp;&nbsp; Colorbar range.

**extend**: {'neither', 'both', 'min', 'max'}

&nbsp;&nbsp;&nbsp;&nbsp; Make pointed end(s) for out-of-range values (unless 'neither') in colorbar.

**colorbar_orientation**: str, optional

&nbsp;&nbsp;&nbsp;&nbsp; Colorbar orientation.

### cycles.plot_yield

`cycles.plot_yield(harvest_df, ax=None, fontsize=None)`

&nbsp;&nbsp;&nbsp;&nbsp; Plot Cycles simulated crop yield.

**harvest_df**: pandas.DataFrame

&nbsp;&nbsp;&nbsp;&nbsp; A pandas DataFrame read from `cycles.read_harvest`.

**ax**: matplotlib.axes.Axes, optional

&nbsp;&nbsp;&nbsp;&nbsp; The axes in which the plot is generated.

**fontsize**: float, optional

### cycles.plot_operations

`cycles.plot_operations(operation_df, rotation_size, ax=None, fontsize=None)`

Visualize operations defined in a management operations file.

**operatin_df**: pandas.DataFrame

&nbsp;&nbsp;&nbsp;&nbsp; A pandas DataFrame read from `cycles.read_operations`.

**rotation_size**: int

&nbsp;&nbsp;&nbsp;&nbsp; Number of years in each rotation.

**ax**: matplotlib.axes.Axes, optional

&nbsp;&nbsp;&nbsp;&nbsp; The axes in which the plot is generated.

**fontsize**: float, optional

## CyclesRunner

Run multiple Cycles simulations defined from a `.csv` file or a pandas DataFrame.
Simulated harvest results are aggregated into a `.csv` file in a `summary` directory.

### Constructor

`CyclesRunner(simulation, summary='summary.csv', simulation_name, control_dict, operation_template=None, operation_dict=None)`

**simulation**: pandas.DataFrame

&nbsp;&nbsp;&nbsp;&nbsp; A pandas DataFrame, each row of which defining a single Cycles simulation.

**summary**: str, optional

&nbsp;&nbsp;&nbsp;&nbsp; Name of summary file.

**simulation_name**: Callable

&nbsp;&nbsp;&nbsp;&nbsp; Function that applies to each row of `simulation` to get the name of each simulation.

**control_dict**: Callable

&nbsp;&nbsp;&nbsp;&nbsp; Function that applies to each row of `simulation` to get the user-defined control parameter dictionary, that can be used for `cycles.generate_control_file` to generate simulation control files.

**operation_template**: str or None, optional

&nbsp;&nbsp;&nbsp;&nbsp; Name of management operations file template that can be used to generate management operations file.

**operation_dict**: dict

&nbsp;&nbsp;&nbsp;&nbsp; Dictionary to be used with `operation_template` for substitution to generate management operations file.

### Run

`CyclesRunner.run(cycles_executable, options, rm_input, rm_output)`

Execute all Cycles simulations.

**cycles_executable**: str

&nbsp;&nbsp;&nbsp;&nbsp; Name (including path) of cycles executable.

**options**: str, optional

&nbsp;&nbsp;&nbsp;&nbsp; Command line options for Cycles (e.g., `-s` for spin-up).

**rm_input**: bool

&nbsp;&nbsp;&nbsp;&nbsp; Whether to remove simulation specific input after simulation.

**rm_output**: bool

&nbsp;&nbsp;&nbsp;&nbsp; Whether to remove simulation specific output after simulation.
