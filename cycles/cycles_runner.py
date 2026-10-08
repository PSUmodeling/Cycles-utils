from __future__ import annotations
import os
import pandas as pd
import shutil
import subprocess
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any
from .cycles import Cycles
from .cycles_tools import generate_control_file, generate_nudge_file
from .cycles_tools.control_file import DEFAULT_CROP_FILE
from .cycles_tools._base_file import _resolve_dict_values, _if_ipython, _disable_progress_bar
if _if_ipython(): from tqdm.notebook import tqdm
else: from tqdm import tqdm

SimulationConfig = list[dict] | pd.DataFrame | list[None] | None

INPUT_DIR: str = 'input'
OUTPUT_DIR: str = 'output'
SUMMARY_DIR: str = 'summary'

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
    crop_dict: dict | None
    operation_dict: dict | None
    calibration_dict: dict | None
    crop_fn: Path
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
    path: Path | str = '.'

    def __post_init__(self):
        self.executable = str(Path(self.executable).resolve())
        self.path = Path(self.path)


    def run(self, *, control_dict: dict[str, Any], simulations: SimulationConfig=None,
            summary: Sequence[str] | None=None, summary_prefix: str | None=None, user_comment: str='',
            operation_template: Path | str | None=None, operation_dict: dict[str, Any] | None=None,
            crop_template: Path | str | None=None, crop_dict: dict[str, Any] | None=None,
            calibration_dict: dict[str, Any] | None=None,
            options: str='',
            rm_input: bool=False, rm_output: bool=False, rm_steady_state_soil: bool=True,
            silence: bool=True,
            _progress_bar: tqdm | None=None) -> None:   # type: ignore
        """Execute a batch of simulations and write a consolidated summary.

        Args:
            simulations: Simulation configurations as list of dicts or a DataFrame. Each dict or DataFrame row should be
                corresponding to a single simulation and contain values to support the control, operation, and
                calibration dictionaries. If None, a single simulation is run directly from `control_dict`/
                `operation_dict` (no per-row substitution). In this single-run mode, the summary CSV is **not**
                written and `rm_input`/`rm_output`/`rm_steady_state_soil` are **not** applied, regardless of how
                those flags are set -- the full input/output is always left in place for inspection.
            control_dict: Control-file values or callables evaluated per simulation.
            summary: List or tuple of output file types to be summarized/aggregated. If None, only the harvest summary is written into `summary/harvest.csv`.
            summary_prefix: Prefix for the summary CSV files. If None, no prefix is added.
            user_comment: Optional text prefixed to summary header comments.
            crop_template: Template file for generated crop files.
            crop_dict: Substitutions used with crop template.
            operation_template: Template file for generated operation files.
            operation_dict: Substitutions used with operation template.
            calibration_dict: Nudge-file values or callables per simulation.
            options: Cycles command options.
            rm_input: Remove generated input files after each run.
            rm_output: Remove run output directory after each run.
            rm_steady_state_soil: Remove generated steady-state soil file.
            silence: If True, suppress simulation screen output.

        In batch mode (`simulations` is a list/DataFrame), a simulation that fails (non-zero exit code) is skipped --
        its summary row is not written -- but does not abort the remaining batch; cleanup (`rm_input`/`rm_output`)
        still runs for the failed row if requested.

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

        The operation dictionary should work with a template operation file to generate the appropriate operation files
        for each simulation. In the template operation file, use placeholders for planting `DOY`, `END_DOY`, and `CROP`
        like below:

        ```
        DOY         $PD1
        END_DOY     $PD2
        CROP        $CROP
        ```

        Then define the operation dictionary to substitute the placeholders with values from the simulation
        configurations:

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
            options='-s',
        )
        ```

        The `-s` option enables spin-up for the simulations. The results will be consolidated into `summary/harvest.csv`.
        """
        if calibration_dict is not None and 'n' not in options:
            warnings.warn('Nudge parameters are provided but Cycles is not running in nudge mode.', UserWarning)

        _check_template_and_dict('crop', crop_template, crop_dict)
        _check_template_and_dict('operation', operation_template, operation_dict)

        crop_template = Path(crop_template) if crop_template is not None else None
        operation_template = Path(operation_template) if operation_template is not None else None
        user_comment = f'# {user_comment.lstrip("# ").rstrip()}\n' if user_comment else ''
        comment = user_comment + _generate_comment(self.executable, options)
        first_run = True

        summary = ['harvest'] if summary is None else list(set(summary) | {'harvest'})

        for s in summary:
            if s == 'harvest': continue
            control_dict[OUTPUT_CONTROL_FLAGS[s]] = 1

        assert isinstance(self.path, Path)
        (self.path / SUMMARY_DIR).mkdir(exist_ok=True)

        simulations = _prepare_simulations(simulations)
        assert simulations is not None

        owns_progress = _progress_bar is None and silence and len(simulations) > 1
        progress = tqdm(simulations, unit='simulation', disable=_disable_progress_bar()) if owns_progress else _progress_bar

        def _report(message: str) -> None:
            tqdm.write(message) if progress is not None else print(message)

        for s in simulations:
            cxt: SimulationContext = self._resolve(s, control_dict, crop_dict, operation_dict, calibration_dict)
            if owns_progress and progress is not None:
                progress.set_description(f'Running {cxt.name}')

            self._write_inputs(cxt, crop_template, operation_template)

            code, _ = _run_cycles_simulation(self.path, self.executable, cxt.name, options, silence)
            if code != 0:
                _report(f'{cxt.name} - Fail (exit code {code})')
            if progress is not None:
                progress.update()

            if s is None:
                return

            if code == 0:
                cycles = Cycles(simulation=cxt.name, path=self.path)
                _write_summary(self.path, cycles, summary, summary_prefix, header=first_run, comment=comment)
                first_run = False
            if rm_input:
                self._remove_inputs(cxt)
            if rm_output:
                shutil.rmtree(self.path / OUTPUT_DIR / cxt.name, ignore_errors=True)
            if rm_steady_state_soil and 's' in options:
                # Steady-state soil should only be removed if generated during this run (i.e., spin-up was requested).
                # If using an existing steady-state soil file, it should not be removed.
                (self.path / INPUT_DIR / f'{cxt.name}_ss.soil').unlink(missing_ok=True)

        if owns_progress and progress is not None:
            progress.set_description('Done')


    def _resolve(self, simulation: dict[str, Any] | None, control_dict: dict[str, Any], crop_dict: dict[str, Any] | None, operation_dict: dict[str, Any] | None,
                 calibration_dict: dict[str, Any] | None) -> SimulationContext:
        control = _resolve_dict_values(control_dict, simulation)
        assert control is not None
        assert isinstance(self.path, Path)
        return SimulationContext(
            name=control['simulation_name'],
            control_dict=control,
            crop_dict=_resolve_dict_values(crop_dict, simulation),
            operation_dict=_resolve_dict_values(operation_dict, simulation),
            calibration_dict=_resolve_dict_values(calibration_dict, simulation),
            operation_fn=self.path / INPUT_DIR / control['operation_file'],
            crop_fn=self.path / INPUT_DIR / control.get('crop_file', DEFAULT_CROP_FILE),
        )


    def _write_inputs(self, cxt: SimulationContext, crop_template: Path | None, operation_template: Path | None) -> None:
        assert isinstance(self.path, Path)
        if crop_template is not None:
            assert cxt.crop_dict is not None
            _render_template(crop_template, cxt.crop_fn, cxt.crop_dict)
        if operation_template is not None:
            assert cxt.operation_dict is not None
            _render_template(operation_template, cxt.operation_fn, cxt.operation_dict)
        if cxt.calibration_dict is not None:
            generate_nudge_file(self.path / INPUT_DIR / f'{cxt.name}.nudge', cxt.calibration_dict)
        generate_control_file(self.path / INPUT_DIR / f'{cxt.name}.ctrl', cxt.control_dict)


    def _remove_inputs(self, cxt: SimulationContext) -> None:
        assert isinstance(self.path, Path)
        (self.path / INPUT_DIR / f'{cxt.name}.ctrl').unlink(missing_ok=True)
        (self.path / INPUT_DIR / f'{cxt.name}.nudge').unlink(missing_ok=True)
        if cxt.crop_dict is not None:
            cxt.crop_fn.unlink(missing_ok=True)
        if cxt.operation_dict is not None:
            cxt.operation_fn.unlink(missing_ok=True)


def _write_summary(path: Path, cycles: Cycles, summary: list[str], summary_prefix: str | None, *, header: bool, comment: str) -> None:
    cycles.read_output(summary)
    for s in summary:
        cycles.output[s].data.insert(0, 'simulation', cycles.simulation)

        mode = 'w' if header else 'a'
        with open(path / SUMMARY_DIR / (f'{summary_prefix}_{s}.csv' if summary_prefix else f'{s}.csv'), mode) as f:
            if header:
                f.write(comment)
            cycles.output[s].data.to_csv(f, header=header, index=False)


def _check_template_and_dict(input_type: str, template_name: Path | str | None, dict_name: dict | None):
    if (template_name is None) != (dict_name is None):
        raise ValueError(
            f"{input_type}_template and {input_type}_dict must be provided together or not at all. "
            f"Got {input_type}_template={'None' if template_name is None else repr(template_name)}, "
            f"{input_type}_dict={'None' if dict_name is None else '...'}"
        )


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


def _run_cycles_simulation(path: Path, executable: str, simulation: str, options: str, silence: bool) -> tuple[int, str]:
    cwd = os.getcwd()
    cmd = [executable, *(options.split() if options else []), simulation]

    os.chdir(path)
    result = subprocess.run(
        cmd,
        shell=os.name == 'nt',
        capture_output=True,
        text=True,
    )
    if not silence:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    os.chdir(cwd)

    return result.returncode, result.stdout


def _prepare_simulations(simulations: SimulationConfig) -> list:
    if simulations is None:
        simulations = [None]
    if isinstance(simulations, pd.DataFrame):
        simulations = simulations.to_dict(orient='records')

    return simulations
