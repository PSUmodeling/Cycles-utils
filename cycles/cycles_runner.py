from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from .cycles import Cycles
from .cycles_tools import generate_control_file, generate_nudge_file, resolve_dict_values

INPUT_DIR: Path = Path('input')
OUTPUT_DIR: Path = Path('output')
SUMMARY_DIR: Path = Path('summary')

@dataclass
class SimulationContext:
    """Holds all resolved values for a single simulation row."""
    name: str
    control_dict: dict
    operation_dict: dict | None
    calibration_dict: dict | None
    operation_fn: Path


@dataclass
class CyclesRunner:
    executable: Path | str
    rotation_builder: bool = False

    def __post_init__(self):
        self.executable = str(Path(self.executable).resolve())


    def run(self, simulations: list[dict], control_dict: dict[str, Any], *,
        summary: str='summary.csv', operation_template: Path | str | None=None, operation_dict: dict[str, Any] | None=None, calibration_dict: dict[str, Any] | None=None,
        options: str='', rm_input: bool=False, rm_output: bool=False, rm_steady_state_soil: bool=True, silence: bool=True, user_comment: str='') -> None:

        if (operation_template is None) != (operation_dict is None):
            raise ValueError(
                "operation_template and operation_dict must be provided together or not at all. "
                f"Got operation_template={'None' if operation_template is None else repr(operation_template)}, "
                f"operation_dict={'None' if operation_dict is None else '...'}"
            )

        if 's' in options and self.rotation_builder:
            raise ValueError('Spin-up cannot be used with rotation builder.')

        operation_template = Path(operation_template) if operation_template is not None else None
        comment = user_comment + _generate_comment(self.executable, options)
        first_run = True

        SUMMARY_DIR.mkdir(exist_ok=True)

        for s in simulations:
            cxt: SimulationContext = self._resolve(s, control_dict, operation_dict, calibration_dict)
            print(f'{cxt.name} - ', end='')

            self._write_inputs(cxt, operation_template)

            if self._run_cycles(cxt.name, options, silence):
                self._write_summary(cxt.name, summary, header=first_run, comment=comment)
                first_run = False
                print('Success')
            else:
                print('Fail')

            if rm_input:
                self._remove_inputs(cxt)
            if rm_output:
                shutil.rmtree(OUTPUT_DIR / cxt.name, ignore_errors=True)
            if rm_steady_state_soil:
                (INPUT_DIR / f'{cxt.name}_ss.soil').unlink(missing_ok=True)

    # -----------------------------------------------------------------------
    # Private — simulation lifecycle
    # -----------------------------------------------------------------------

    def _resolve(self, simulation: dict[str, Any], control_dict: dict[str, Any], operation_dict: dict[str, Any] | None, calibration_dict: dict[str, Any] | None) -> SimulationContext:
        """Call control_dict once per row and resolve all derived values."""
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
            _render_template(operation_template, cxt.operation_fn, cxt.operation_dict)
        if cxt.calibration_dict is not None:
            generate_nudge_file(INPUT_DIR / f'{cxt.name}.nudge', cxt.calibration_dict)
        generate_control_file(INPUT_DIR / f'{cxt.name}.ctrl', cxt.control_dict, rotation_builder=self.rotation_builder)


    def _remove_inputs(self, cxt: SimulationContext) -> None:
        (INPUT_DIR / f'{cxt.name}.ctrl').unlink(missing_ok=True)
        (INPUT_DIR / f'{cxt.name}.nudge').unlink(missing_ok=True)
        if cxt.operation_dict is not None:
            cxt.operation_fn.unlink(missing_ok=True)


    def _write_summary(self, name: str, summary: str, *, header: bool, comment: str) -> None:
        cycles = Cycles(path='.', simulation=name)
        cycles.read_output('harvest')
        cycles.output['harvest'].data.insert(0, 'simulation', name)

        mode = 'w' if header else 'a'
        with open(SUMMARY_DIR / summary, mode) as f:
            if header:
                f.write(comment)
            cycles.output['harvest'].data.to_csv(f, header=header, index=False)

    # -----------------------------------------------------------------------
    # Private — subprocess helpers
    # -----------------------------------------------------------------------

    def _run_cycles(self, simulation: str, options: str, silence: bool) -> bool:
        cmd = [self.executable, *(options.split() if options else []), simulation]
        result = subprocess.run(
            cmd,
            shell=os.name == 'nt',
            stdout=subprocess.DEVNULL if silence else None,
            stderr=subprocess.DEVNULL if silence else None,
        )
        return result.returncode == 0


# ---------------------------------------------------------------------------
# Module-level helpers — pure functions with no dependency on CyclesRunner
# ---------------------------------------------------------------------------

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
        'with calibration' if 'c' in options else None,
        'grain model turned on' if 'g' in options else None,
    ]
    return ', '.join(p for p in parts if p) + '\n'
