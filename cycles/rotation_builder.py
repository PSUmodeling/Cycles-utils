from __future__ import annotations
import numpy as np
import pandas as pd
import re
from dataclasses import dataclass, InitVar, field, replace
from itertools import product
from pathlib import Path
from .cycles_runner import CyclesRunner
from .cycles import Cycles
from .cycles_tools import Operation, Planting, Tillage, Harvest, Kill, FixedFertilization, FixedIrrigation, AutoIrrigation
from .cycles_tools import generate_control_file

MIN_PLANTING_INTERVAL = 7

@dataclass(frozen=True)
class CropGroup:
    name: str
    crops: frozenset[str]
    penalty_factor: float

CROP_GROUPS: tuple[CropGroup, ...] = (
    CropGroup('BRASSICA', frozenset({'THAR5'}), 0.075),
    CropGroup('DICOT_LEGUME', frozenset({'GLMA4'}), 0.103),
    CropGroup('DICOT_NONLEGUME', frozenset({''}), 0.089),
    CropGroup('GRASS', frozenset({'ZEMA', 'TRAE'}), 0.043),
)

@dataclass
class Crop:
    name: str
    symbol: str
    operations: list[Planting | Tillage | FixedFertilization]
    times_planted: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        for op in self.operations:
            if isinstance(op, Planting) and op.crop is None:
                op.crop = self.name
            if hasattr(op, 'year'):
                op.year = 1

@dataclass(frozen=True)
class EconomicParameters:
    crop_price: dict | None
    fertilizer_price: dict | None
    production_cost: dict | None
    penalty_factors: dict[tuple[str, str], float]  # (crop1.name, crop2.name) -> factor

    @classmethod
    def from_builder(cls, builder: CyclesRotationBuilder, crops: list[Crop], year: int) -> EconomicParameters:
        penalty_factors = {(crop1.name, crop2.name): _calculate_penalty_factor(crop1, crop2) for crop1 in crops for crop2 in crops}

        return cls(
            crop_price=builder.crop_price_data.loc[year].to_dict() if builder.crop_price_data is not None else None,
            fertilizer_price=builder.fertilizer_price_data.loc[year].to_dict() if builder.fertilizer_price_data is not None else None,
            production_cost=builder.production_cost_data.loc[year].to_dict() if builder.production_cost_data is not None else None,
            penalty_factors=penalty_factors,
        )


@dataclass
class CyclesRotationBuilder:
    simulation: str
    executable: str
    crops: list[Crop]
    control_dict: dict
    crop_price: InitVar[str | Path |  None] = field(default=None, kw_only=True)
    fertilizer_price: InitVar[str | Path |  None] = field(default=None, kw_only=True)
    production_cost: InitVar[str | Path |  None] = field(default=None, kw_only=True)
    yield_matrix: dict[str, pd.DataFrame] = field(init=False)
    crop_price_data: pd.DataFrame | None = field(init=False, default=None)
    fertilizer_price_data: pd.DataFrame | None = field(init=False, default=None)
    production_cost_data: pd.DataFrame | None = field(init=False, default=None)
    rotation_frequency: dict[str, tuple[float, float]] | None = field(default=None, kw_only=True)
    build_yield_matrix: bool = field(default=True, kw_only=True)

    def __post_init__(self, crop_price: str | Path | None, fertilizer_price: str | Path | None, production_cost: str | Path | None) -> None:
        self.executable = str(Path(self.executable).resolve())

        if self.build_yield_matrix:
            self._build_yield_matrix()
        else:
            self.yield_matrix = {crop.name: self._read_yield_matrix(crop) for crop in self.crops}

        self.crop_price_data = _optional_csv(crop_price)
        self.fertilizer_price_data = _optional_csv(fertilizer_price)
        self.production_cost_data = _optional_csv(production_cost)



    def run(self) -> None:
        operations: list[Operation] = []
        self.control_dict['rotation_size'] = int(self.control_dict['simulation_end_date']) - int(self.control_dict['simulation_start_date']) + 1
        self.control_dict['operation_file'] = f'{self.simulation}.operation'

        penalty_factors = {(crop1.name, crop2.name): _calculate_penalty_factor(crop1, crop2) for crop1 in self.crops for crop2 in self.crops}

        generate_control_file(f'./input/{self.simulation}.ctrl', self.control_dict)
        _write_operation_file(Path('./input') / f'{self.simulation}.operation', operations)

        cycles = Cycles(path='.', simulation=self.simulation, executable=self.executable)

        options = '-b'
        while (True):
            status, screen_output = cycles.run(options=options, silence=False)
            if status != 10:
                break
            options = '-cb'
            year, doy = _find_break_doy(screen_output)
            print(f'Break point reached at Year {year} DOY {doy}.')

            economic_parameters = EconomicParameters.from_builder(self, self.crops, year)

            last_crop = _find_crop(self.crops, _last_planting(operations).crop) if operations else None # type: ignore

            self._build_rotation(year, doy, last_crop, economic_parameters, operations)

            _write_operation_file(Path('./input') / f'{self.simulation}.operation', operations)



    def _build_yield_matrix(self) -> None:
        self.yield_matrix = {}
        _write_operation_templates(self.crops)

        cycles_runner = CyclesRunner(self.executable)
        for crop in self.crops:
            simulations, control_dict, operation_dict = _build_simulations(crop.operations, self.control_dict)
            cycles_runner.run(
                simulations=simulations,
                summary=f'{crop.name}.csv',
                control_dict=control_dict,
                operation_dict=operation_dict,
                operation_template=f'template/{crop.name}.operation',
                silence=True,
                rm_input=True,
                rm_output=True,
            )
            self.yield_matrix[crop.name] = self._read_yield_matrix(crop)


    def _build_rotation(self, year: int, doy: int, last_crop: Crop | None, economic_parameters: EconomicParameters, operations: list[Operation]) -> None:
        best_return = float('-inf')
        best_crop: Crop
        best_doy: int

        for crop1 in self.crops:
            if last_crop and crop1.symbol == last_crop.symbol == 'GLMA4': continue
            planting1 = _last_planting(crop1.operations)
            penalty1 = economic_parameters.penalty_factors.get((last_crop.name, crop1.name), 0.0) if last_crop else 0.0
#            adjust = FreqAdjust(rot_year, mgmt->planting[icrop].crop->epc.symbol, mgmt);
            for crop2 in self.crops:
                if crop1.symbol == crop2.symbol == 'GLMA4': continue
                planting2 = _last_planting(crop2.operations)
                penalty2 = economic_parameters.penalty_factors.get((crop1.name, crop2.name), 0.0)

                for doy1 in range(planting1.doy, planting1.end_doy + 1):    # type: ignore
                    for doy2 in range(planting2.doy, planting2.end_doy + 1):    # type: ignore
                        ret = self._calculate_economic_return(year, doy, doy1, doy2, crop1, crop2, penalty1, penalty2, economic_parameters)
                        if ret > best_return:
                            best_return, best_crop, best_doy = ret, crop1, doy1

        planting_year = year - int(self.control_dict['simulation_start_date']) + 1 if best_doy > doy else year - int(self.control_dict['simulation_start_date']) + 2

        for op in best_crop.operations:
            assert(op.doy is not None)
            if isinstance(op, Planting):
                operations.append(replace(op, year=planting_year, doy=best_doy))
            elif isinstance(op, (Tillage, FixedFertilization)):
                if op.doy < 0:
                    op_doy = best_doy + op.doy
                    op_year = planting_year
                else:
                    op_doy = (best_doy + op.doy) % 366
                    op_year = planting_year + (best_doy + op.doy) // 366
                operations.append(replace(op, year=op_year, doy=op_doy))
            else:
                operations.append(op)

        best_crop.times_planted += 1


    def _calculate_economic_return(self, year: int, doy: int, doy1: int, doy2: int, crop1: Crop, crop2: Crop, penalty1: float, penalty2: float, economic_parameters: EconomicParameters) -> float:
        def _sample_yield(crop: Crop, plant_doy: int) -> pd.Series:
            rows = self.yield_matrix[crop.name]
            return rows.loc[rows['doy'] == plant_doy].sample().iloc[0]

        row1 = _sample_yield(crop1, doy1)
        row2 = _sample_yield(crop2, doy2)

        total_days = _calculate_total_window(doy, doy1, row1['growing_window'], doy2, row2['growing_window'])

        if self.crop_price_data is None:
            total_income = (row1['grain_yield'] + row1['forage_yield']) * penalty1 + (row2['grain_yield'] + row2['forage_yield']) * penalty2
            return total_income / total_days

        def _n_rate(crop: Crop, row: pd.Series) -> float:
            return 0.0 if _find_crop_group(crop.symbol).name == 'DICOT_LEGUME' else row['nitrogen_in_harvest'] * 1.33

        total_income = ((row1['grain_yield'] * economic_parameters.crop_price[crop1.symbol] + row1['forage_yield'] * economic_parameters.crop_price[crop1.symbol]) * (1.0 - penalty1) +
            (row2['grain_yield'] * economic_parameters.crop_price[crop2.symbol] + row2['forage_yield'] * economic_parameters.crop_price[crop2.symbol]) * (1.0 - penalty2))
        if self.production_cost_data is not None:
            total_income -= economic_parameters.production_cost[crop1.symbol] + economic_parameters.production_cost[crop2.symbol]
        if self.fertilizer_price_data is not None:
            total_income -= (_n_rate(crop1, row1) * economic_parameters.fertilizer_price['urea'] + _n_rate(crop2, row2) * economic_parameters.fertilizer_price['urea'])

        return total_income / total_days




    def _read_yield_matrix(self, crop: Crop) -> pd.DataFrame:
        df = pd.read_csv(f'summary/{crop.name}.csv', usecols=[1, 3, 6, 7, 18], comment='#')
        df['date'] = pd.to_datetime(df['date'])
        df['planting_date'] = pd.to_datetime(df['planting_date'])
        df['doy'] = pd.to_datetime(df['planting_date']).dt.dayofyear
        df['year'] = pd.to_datetime(df['planting_date']).dt.year
        df['growing_window'] = (df['date'] - df['planting_date']).dt.days
        df.drop(columns=['date', 'planting_date'], inplace=True)

        start_year = self.control_dict['simulation_start_date']
        end_year = self.control_dict['simulation_end_date']
        df = df[(df['year'] != start_year) & (df['year'] != end_year)].copy()

        planting = _last_planting(crop.operations)
        existing = set(zip(df['year'], df['doy']))

        missing_rows = [
            {'year': y, 'doy': d, 'growing_window': 365, 'grain_yield': 0.0, 'forage_yield': 0.0, 'nitrogen_in_harvest': 0.0}
            for y in range(start_year + 1, end_year)
            for d in range(planting.doy, planting.end_doy + 1)
            if (y, d) not in existing
        ]
        if missing_rows:
            df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)

        return df


def _build_simulations(operations: list[Operation], user_dict: dict) -> tuple[list[dict], dict, dict]:
    simulations: list[dict] = []
    operation_dict: dict = {}
    planting = _last_planting(operations)

    for ind, op in enumerate(operations):
        key = f'DOY{ind + 1}'
        operation_dict[key] = lambda x, k=key: x[k]

    for doy in range(planting.doy, planting.end_doy + 1):
        sim: dict = {'simulation_name': f'{planting.crop}_{doy}'}
        for ind, op in enumerate(operations):
            sim[f'DOY{ind + 1}'] = (
                doy if isinstance(op, Planting)
                else _day_of_year(doy + op.doy) if op.doy < 0
                else f'+{op.doy}'
            )
        simulations.append(sim)

    control_dict = user_dict | {
        'simulation_name': lambda x: x['simulation_name'],
        'rotation_size': 1,
        'operation_file': f'{planting.crop}.operation',
        'automatic_nitrogen': 1,
    }

    return simulations, control_dict, operation_dict


def _last_planting(operations: list) -> Planting:
    planting = next((op for op in reversed(operations) if isinstance(op, Planting)), None)
    if planting is None:
        raise ValueError('No planting operation found in operations list.')
    return planting


def _find_crop(crops: list[Crop], name: str) -> Crop:
    crop = next((crop for crop in crops if crop.name == name), None)
    if crop is None:
        raise ValueError(f'Crop name {name} not found in crops list.')
    return crop


def _find_crop_group(crop_symbol: str) -> CropGroup:
    group = next((group for group in CROP_GROUPS if crop_symbol in group.crops), None)
    if group is None:
        raise ValueError(f'Crop symbol {crop_symbol} not found in any crop group.')
    return group


def _find_break_doy(output: str) -> tuple[int, int]:
    """Return (year, doy) of the first break point, or None if not found."""
    match = re.search(r'Break point reached.*?Year (\d+) DOY (\d+)', output)
    if match is None:
        raise ValueError('No break point found in output.')
    return int(match.group(1)), int(match.group(2))

def _calculate_penalty_factor(crop1: Crop | None, crop2: Crop | None) -> float:
    if crop1 is None or crop2 is None:
        return 0.0
    group1 = _find_crop_group(crop1.symbol)
    group2 = _find_crop_group(crop2.symbol)
    return group1.penalty_factor if group1 is group2 else 0.0


def _calculate_total_window(doy: int, doy1: int, window1: int, doy2: int, window2: int) -> int:
    total_days = doy1 - doy + 365 * (doy + MIN_PLANTING_INTERVAL > doy1)
    total_days += window1 + MIN_PLANTING_INTERVAL
    temp_doy = (doy1 + window1 + MIN_PLANTING_INTERVAL) % 365
    total_days += doy2 - temp_doy + 365 * (doy2 < temp_doy)
    total_days += window2

    return total_days


def _camel_to_snake(text: str) -> str:
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text).lower()


def _day_of_year(day: int) -> int:
    return day if day <= 365 else day - 365


def _optional_csv(path: Path | str | None) -> pd.DataFrame | None:
    return pd.read_csv(path, index_col=0) if path is not None else None


def _format_operation(operation: Operation, doy_override: dict[str, str] | None = None) -> list[str]:
    """Serialise a single operation to lines, with optional DOY substitutions."""
    lines = [_camel_to_snake(type(operation).__name__).upper()]
    for key, val in operation.__dict__.items():
        if doy_override and key in doy_override:
            val = doy_override[key]
        lines.append(f'{key.upper():<20}{val}')
    lines.append('')
    return lines

def _write_operation_templates(crops: list[Crop]) -> None:
    for crop in crops:
        path = Path('template') / f'{crop.name}.operation'
        path.parent.mkdir(exist_ok=True)

        lines: list[str] = []
        for ind, op in enumerate(crop.operations):
            overrides = {'doy': f'$DOY{ind + 1}'}
            if isinstance(op, Planting):
                overrides['end_doy'] = '-999'
            lines.extend(_format_operation(op, overrides))

        path.write_text('\n'.join(lines))


def _write_operation_file(path: Path, operations: list[Operation]) -> None:
    lines: list[str] = []
    for op in operations:
        lines.extend(_format_operation(op))
    path.write_text('\n'.join(lines))
