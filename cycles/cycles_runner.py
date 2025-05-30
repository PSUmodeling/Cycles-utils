import os
import pandas as pd
import subprocess
from string import Template
from typing import Callable
from .cycles_input import generate_control_file
from .cycles_read import read_harvest

RM_CYCLES_IO = 'rm -fr input/*.ctrl input/*.operation input/*.soil output/*'

class CyclesRunner():
    def __init__( self, *, simulations: pd.DataFrame | str, summary: str='summary.csv', simulation_name: Callable, control_dict: Callable, operation_template: str | None=None, operation_dict: Callable | None=None):
        if type(simulations) is str:
            self.simulations = pd.read_csv(simulations)
        elif type(simulations) is pd.DataFrame:
            self.simulations = simulations
        else:
            raise TypeError('simulations must be a DataFrame or a path to a CSV file')
        self.summary_file = summary
        self.operation_template = operation_template
        self.operation_dict = operation_dict
        self.simulation_name = simulation_name
        self.control_dict = control_dict


    def run(self, cycles_executable: str, *, spin_up: bool=True):
        """Run Cycles simulations as defined
        """
        os.makedirs('summary', exist_ok=True)

        # Read simulation file
        first = True
        for _, row in self.simulations.iterrows():
            # Use defined simulation name
            name = self.simulation_name(row)

            print(f'{name} - ', end='')

            # Get name of operation file from defined control parameters
            operation_fn = self.control_dict(row)['operation_file']

            # Create operation file if needed
            if self.operation_template is not None and self.operation_dict is not None:
                _generate_input_from_template(self.operation_template, f'./input/{operation_fn}', self.operation_dict(row))

            # Generate control file from defined control parameters
            generate_control_file(f'./input/{name}.ctrl', self.control_dict(row))

            # Run a Cycles simulation
            if _run_cycles(cycles_executable, name, spin_up=spin_up) == 0:
                _write_summary(name, first, f'summary/{self.summary_file}')
                print('Success')
                first = False
            else:
                print('Fail')

            # Remove generated input/output files
            subprocess.run(
                RM_CYCLES_IO,
                shell='True',
            )


def _run_cycles(cycles_executable, simulation: str, spin_up: bool=True):
    cmd = [cycles_executable, simulation] if spin_up is False else [cycles_executable, '-s', simulation]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return result.returncode


def _write_summary(simulation: str, header: bool, summary_fn: str) -> None:
    df = read_harvest('.', simulation)
    df.insert(0, 'simulation', simulation)

    if header is True:
        with open(summary_fn, 'w') as f:
            df.to_csv(f, header=True, index=False)
    else:
        with open(summary_fn, 'a') as f:
            df.to_csv(f, header=False, index=False)


def _generate_input_from_template(template_fn: str, input_fn: str, user_dict: dict) -> None:
    with open(template_fn) as f:
        operation_file_template = Template(f.read())

    with open(input_fn, 'w') as f:
        f.write(operation_file_template.substitute(user_dict) + '\n')
