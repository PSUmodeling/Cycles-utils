from __future__ import annotations
import os
import pandas as pd
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any
from .cycles import Cycles
from .cycles_tools import generate_control_file, generate_nudge_file, resolve_dict_values

SimulationConfig = list[dict] | pd.DataFrame

INPUT_DIR: Path = Path('input')
OUTPUT_DIR: Path = Path('output')
SUMMARY_DIR: Path = Path('summary')

OUTPUT_CONTROL_FLAGS: dict = {
    'dailyEnviron': 'daily_weather_out',
    'dailyResidue': 'daily_residue_out',
    'dailyWater': 'daily_water_out',
    'dailyN': 'daily_nitrogen_out',
    'dailySoilC': 'daily_soil_carbon_out',
    'dailySoilLayersCN': 'daily_soil_lyr_cn_out',
    'annualSOM': 'annual_soil_out',
    'annualSoilProfileC': 'annual_profile_out',
    'annualN': 'annual_nflux_out',
}

@dataclass
class SimulationContext:
    name: str
    control_dict: dict
    operation_dict: dict | None
    calibration_dict: dict | None
    operation_fn: Path


@dataclass
class CyclesRunner:
    """Run one or many Cycles simulations with templated inputs.

    Manages batch execution of Cycles simulations by generating control files, operation files, and nudge files from
    templates and parameter dictionaries.  Consolidates results into a summary CSV file.

    Args:
        executable: Absolute path to the Cycles executable binary.
    """

    executable: str

    def __post_init__(self):
        self.executable = str(Path(self.executable).resolve())


    def run(self, simulations: SimulationConfig, control_dict: dict[str, Any], *,
        summary: str | dict[str, str] | None=None,
        operation_template: Path | str | None=None, operation_dict: dict[str, Any] | None=None,
        calibration_dict: dict[str, Any] | None=None,
        options: str='', rm_input: bool=False, rm_output: bool=False, rm_steady_state_soil: bool=True, silence: bool=True, user_comment: str='') -> None:
        """Execute a batch of simulations and write a consolidated summary.

        Args:
            simulations: Simulation configurations as list of dicts or a DataFrame. Each dict or DataFrame row should be
                corresponding to a single simulation and contain values to support the control, operation, and
                calibration dictionaries.
            control_dict: Control-file values or callables evaluated per simulation.
            summary: Summary CSV name for the summary harvest file written under summary directory. If a dictionary is provided, the keys are output file types and the values are
                summary CSV names. If None, only the harvest summary is written into `summary/summary.csv`.
            operation_template: Template file for generated operation files.
            operation_dict: Substitutions used with operation template.
            calibration_dict: Nudge-file values or callables per simulation.
            options: Cycles command options.
            rm_input: Remove generated input files after each run.
            rm_output: Remove run output directory after each run.
            rm_steady_state_soil: Remove generated steady-state soil file.
            silence: If True, suppress simulation screen output.
            user_comment: Optional text prefixed to summary header comments.

        The following fields are required in `control_dict`:

        - `simulation_name`
        - `simulation_start_year`
        - `simulation_end_year`
        - `rotation_size`
        - `operation_file`
        - `soil_file`
        - `weather_file`

        The default values for other fields are:

        - `crop_file`: `GenericCrops.crop`
        - `reinit_file`: `N/A`
        - `soil_layers`: inferred from the soil file (if not provided)
        - `co2_level`: `-999`
        - `use_reinitialization`: `0`
        - `adjusted_yields`: `0`
        - `hydrology_option`: `1`
        - `automatic_nitrogen`: `0`
        - `automatic_phosphorus`: `0`
        - `automatic_sulfur`: `0`

        All output control fields default to `0`.

        Note that `simulation_name` is used to generate the control file name for each simulation. The `simulation_name`
        should be unique for each simulation in the batch.

        #### Example:

        To run a batch simulation of continuous corn in different counties of Iowa, you can use the following code snippet:
        ```python
        from cycles import CyclesRunner

        runner = CyclesRunner(executable='/path/to/Cycles')

        simulations: list[dict] = [
            'GID': 'USA.16.1_1', 'weather': 'NLDAS_41.438Nx94.562W', 'soil': 'maize_rainfed_SoilGrids_USA.16.1_1.soil', 'plant_start': 112, 'plant_end': 154, 'maturity_group': 100,
            'GID': 'USA.16.2_1', 'weather': 'NLDAS_40.938Nx94.688W', 'soil': 'maize_rainfed_SoilGrids_USA.16.2_1.soil', 'plant_start': 112, 'plant_end': 154, 'maturity_group': 100,
            'GID': 'USA.16.3_1', 'weather': 'NLDAS_43.188Nx91.562W', 'soil': 'maize_rainfed_SoilGrids_USA.16.3_1.soil', 'plant_start': 112, 'plant_end': 154, 'maturity_group': 90,
        ]
        ```

        The control dictionary should work with the simulation configurations to generate the appropriate control files for each simulation:

        ```python
        control_dict: dict = {
            'simulation_name': lambda x: x['GID'],
            'simulation_start_year': 1981,
            'simulation_end_year': 2016,
            'rotation_size': 1,
            'crop_file': 'GenericCrops.crop',
            'operation_file': lambda x: f'{x["GID"]}.operation',
            'soil_file': lambda x: f'path/to/{x["soil"]}',
            'weather_file': lambda x: f'path/to/{x["gridMET_weather"]}.weather',
        }
        ```

        The operation dictionary should work with a template operation file to generate the appropriate operation files for each simulation. In the template operation file, use
        placeholders for planting `DOY`, `END_DOY`, and `CROP` like below:

        ```
        DOY         $PD1
        END_DOY     $PD2
        CROP        $CROP
        ```

        Then define the operation dictionary to substitute the placeholders with values from the simulation configurations:

        ```python
        operation_dict: dict = {
            'PD1': lambda x: x['plant_start'],
            'PD2': lambda x: x['plant_end'],
            'CROP': lambda x: f'CornRM.{x["relative_maturity_group"]}',
        }
        ```

        Finally, run the simulations with the following code snippet:

        ```python
        cycles_runner.run(
            simulations=simulations,
            control_dict=control_dict,
            operation_template='path/to/template.operation',
            operation_dict=operation_dict,
            summary='summary.csv',
            options='-s',
        )
        ```

        The `-s` option enables spin-up for the simulations. The results will be consolidated into `summary/summary.csv`.
        """
        if isinstance(simulations, pd.DataFrame):
            simulations = simulations.to_dict(orient='records')
        assert isinstance(simulations, list)

        if (operation_template is None) != (operation_dict is None):
            raise ValueError(
                "operation_template and operation_dict must be provided together or not at all. "
                f"Got operation_template={'None' if operation_template is None else repr(operation_template)}, "
                f"operation_dict={'None' if operation_dict is None else '...'}"
            )

        operation_template = Path(operation_template) if operation_template is not None else None
        if user_comment:
            user_comment = f'# {user_comment.lstrip("# ").rstrip()}\n'
        comment = user_comment + _generate_comment(self.executable, options)
        first_run = True

        if summary is None:
            summary = {'harvest': 'summary.csv'}
        elif isinstance(summary, str):
            summary = {'harvest': summary}
        assert isinstance(summary, dict)

        for key in summary.keys():
            if key == 'harvest': continue
            control_dict[OUTPUT_CONTROL_FLAGS[key]] = 1

        SUMMARY_DIR.mkdir(exist_ok=True)

        for s in simulations:
            cxt: SimulationContext = self._resolve(s, control_dict, operation_dict, calibration_dict)
            print(f'{cxt.name} - ', end='')

            self._write_inputs(cxt, operation_template)

            cycles = Cycles(path='.', simulation=cxt.name, executable=self.executable)

            code, _ = cycles.run(options=options, silence=silence)

            if code == 0:
                self._write_summary(cycles, summary, header=first_run, comment=comment)
                first_run = False
                print('Success')
            elif code == 1:
                print('Fail')

            if rm_input:
                self._remove_inputs(cxt)
            if rm_output:
                shutil.rmtree(OUTPUT_DIR / cxt.name, ignore_errors=True)
            if rm_steady_state_soil and 's' in options:
                # Steady-state soil should only be removed if generated during this run (i.e., spin-up was requested).
                # If using an existing steady-state soil file, it should not be removed.
                (INPUT_DIR / f'{cxt.name}_ss.soil').unlink(missing_ok=True)


    def _resolve(self, simulation: dict[str, Any], control_dict: dict[str, Any], operation_dict: dict[str, Any] | None, calibration_dict: dict[str, Any] | None) -> SimulationContext:
        control = resolve_dict_values(control_dict, simulation)
        return SimulationContext(
            name=control['simulation_name'],
            control_dict=control,
            operation_dict=resolve_dict_values(operation_dict, simulation) if operation_dict is not None else None,
            calibration_dict=resolve_dict_values(calibration_dict, simulation) if calibration_dict is not None else None,
            operation_fn=INPUT_DIR / control['operation_file'],
        )


    def _write_inputs(self, cxt: SimulationContext, operation_template: Path | None) -> None:
        if operation_template is not None:
            assert cxt.operation_dict is not None
            _render_template(operation_template, cxt.operation_fn, cxt.operation_dict)
        if cxt.calibration_dict is not None:
            generate_nudge_file(INPUT_DIR / f'{cxt.name}.nudge', cxt.calibration_dict)
        generate_control_file(INPUT_DIR / f'{cxt.name}.ctrl', cxt.control_dict)


    def _remove_inputs(self, cxt: SimulationContext) -> None:
        (INPUT_DIR / f'{cxt.name}.ctrl').unlink(missing_ok=True)
        (INPUT_DIR / f'{cxt.name}.nudge').unlink(missing_ok=True)
        if cxt.operation_dict is not None:
            cxt.operation_fn.unlink(missing_ok=True)


    def _write_summary(self, cycles: Cycles, summary: dict, *, header: bool, comment: str) -> None:
        cycles.read_output(summary.keys())
        for key, fn in summary.items():
            cycles.output[key].data.insert(0, 'simulation', cycles.simulation)

            mode = 'w' if header else 'a'
            with open(SUMMARY_DIR / fn, mode) as f:
                if header:
                    f.write(comment)
                cycles.output[key].data.to_csv(f, header=header, index=False)


def _render_template(template_fn: Path, dest_fn: Path, substitutions: dict) -> None:
    dest_fn.write_text(Template(template_fn.read_text()).substitute(substitutions) + '\n')


def _generate_comment(executable: str, options: str) -> str:
    result = subprocess.run(
        [executable, '-V'],
        shell=os.name == 'nt',
        capture_output=True,
        text=True,
    )
    version = ''.join(result.stdout.splitlines())
    parts = [
        f'# {version}',
        'with spin-up' if 's' in options else 'without spin-up',
        'with calibration' if 'n' in options else None,
        'grain model turned on' if 'g' in options else None,
        'dynamically reduced fertilization rates' if 'x' in options else None,
    ]
    return ', '.join(p for p in parts if p) + '\n'
