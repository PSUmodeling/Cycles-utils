from __future__ import annotations
import numpy as np
import pandas as pd
import re
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import NamedTuple
from .cycles import Cycles
from .cycles_runner import CyclesRunner
from .cycles_tools import Operation, Planting, Tillage, FixedFertilization
from .cycles_tools import generate_control_file, format_operation, generate_operation_file

FERTILIZER_FILE = 'input/fertilizers.txt'
MIN_PLANTING_INTERVAL = 7
BREAK_POINT_REACHED = 10

@dataclass
class Fertilizer:
    name: str
    c_organic: float
    c_charcoal: float
    n_organic: float
    n_charcoal: float
    n_nh4: float
    n_no3: float
    p_organic: float
    p_charcoal: float
    p_inorganic: float
    k: float
    s: float

    @property
    def n_fraction(self) -> float:
        return self.n_organic + self.n_charcoal + self.n_nh4 + self.n_no3

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
    operations: list[Operation]
    prescribed_n_rate: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        for op in self.operations:
            if isinstance(op, Planting):
                op.crop = self.name
                op.use_auto_fert = 1
            if hasattr(op, 'year'):
                op.year = 1

@dataclass(frozen=True)
class EconomicParameters:
    crop_price: dict
    fertilizer_price: dict | None
    production_cost: dict | None
    penalty_factors: dict[tuple[str, str], float]  # (crop1.name, crop2.name) -> factor
    yield_matrix: dict[str, pd.DataFrame]

    @classmethod
    def from_builder(cls, builder: CyclesRotationBuilder, year: int) -> EconomicParameters:
        penalty_factors = {(crop1.name, crop2.name): _calculate_penalty_factor(crop1, crop2) for crop1 in builder.crops for crop2 in builder.crops}
        assert builder.crop_price_data is not None
        return cls(
            crop_price=builder.crop_price_data.loc[year].to_dict(),
            fertilizer_price=builder.fertilizer_price_data.loc[year].to_dict() if builder.fertilizer_price_data is not None else None,
            production_cost=builder.production_cost_data.loc[year].to_dict() if builder.production_cost_data is not None else None,
            penalty_factors=penalty_factors,
            yield_matrix=builder.yield_matrix,
        )

class RotationResult(NamedTuple):
    crop: Crop
    doy: int
    n_rate: float | None
    economic_return: float


@dataclass
class CyclesRotationBuilder:
    """Run Cycles iteratively and append economically optimal operations.

    Executes Cycles simulations for candidate crop-operation combinations, selects the highest-return option at each
    break-point, and iteratively builds a multi-year rotation. Integrates economic scoring with agronomic constraints
    (e.g., minimum planting interval, crop group penalties).

    Attributes:
        simulation: Base simulation name for generated input/output directories.
        executable: Absolute path to the Cycles executable binary.
        crops: List of Crop objects with available operations for rotation.
        control_dict: Base control file parameters (simulation years, options, etc.).
        build_yield_matrix: Optional flag: If True, run simulations to build yield matrix; else read from disk.

    Attributes:
        simulation: Base simulation name for generated input/output directories.
        executable: Absolute path to the Cycles executable binary.
        crops: List of Crop objects with available operations for rotation.
        control_dict: Base control file parameters (simulation years, options, etc.).
        fertilizers: Dictionary mapping fertilizer names to Fertilizer objects.
        yield_matrix: Dictionary mapping crop names to yield prediction DataFrames.
        build_yield_matrix: If True, run simulations to build yield matrix; else read from disk.
        crop_price_data: DataFrame of crop prices indexed by calendar year.
        fertilizer_price_data: DataFrame of fertilizer prices indexed by year, or None.
        production_cost_data: DataFrame of production costs indexed by year, or None.
        rotation_frequency: Dictionary mapping crop names to (min, max) frequency tuples.
    """

    simulation: str
    executable: str
    crops: list[Crop]
    control_dict: dict
    fertilizers: dict[str, Fertilizer] = field(init=False, default_factory=dict)
    yield_matrix: dict[str, pd.DataFrame] = field(init=False)
    build_yield_matrix: bool = field(default=True, kw_only=True)
    crop_price_data: pd.DataFrame | None = field(init=False, default=None)
    fertilizer_price_data: pd.DataFrame | None = field(init=False, default=None)
    production_cost_data: pd.DataFrame | None = field(init=False, default=None)
    rotation_frequency: dict[str, tuple[float, float]] | None = field(init=False, default=None)
    _times_planted: dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.executable = str(Path(self.executable).resolve())
        self.fertilizers = _read_fertilizer_file(Path('./') / FERTILIZER_FILE)
        self._times_planted = {crop.symbol: 0 for crop in self.crops}

        for crop in self.crops:
            crop.prescribed_n_rate = sum(op.mass * self.fertilizers[op.source].n_fraction for op in crop.operations if isinstance(op, FixedFertilization))

        if self.build_yield_matrix:
            self._build_yield_matrix()
        else:
            self.yield_matrix = {crop.name: self._read_yield_matrix(crop) for crop in self.crops}


    def run(self, *, crop_price: str | Path, fertilizer_price: str | Path | None=None, production_cost: str | Path | None=None, rotation_frequency: dict[str, tuple[float, float]] | None=None) -> None:
        """Run the dynamic rotation-building loop.

        Args:
            crop_price: CSV file of crop prices indexed by year.
            fertilizer_price: Optional fertilizer price table indexed by year.
            production_cost: Optional crop production cost table indexed by year.
            rotation_frequency: Optional min/max frequency constraints per crop symbol.
        """
        self.crop_price_data = _optional_csv(crop_price)
        self.fertilizer_price_data = _optional_csv(fertilizer_price)
        self.production_cost_data = _optional_csv(production_cost)
        self.rotation_frequency = rotation_frequency

        if self.crop_price_data is None:
            raise ValueError('Crop price data is required to run the rotation builder.')

        operations: list[Operation] = []
        self.control_dict['rotation_size'] = self.control_dict['simulation_end_year'] - self.control_dict['simulation_start_year'] + 1
        self.control_dict['operation_file'] = f'{self.simulation}.operation'

        generate_control_file(f'./input/{self.simulation}.ctrl', self.control_dict)
        generate_operation_file(Path('./input') / f'{self.simulation}.operation', operations)

        cycles = Cycles(path='.', simulation=self.simulation, executable=self.executable)
        options = '-b'

        while True:
            status, screen_output = cycles.run(options=options, silence=False)
            if status != BREAK_POINT_REACHED:
                break

            options = '-rb'
            year, doy = _find_break_doy(screen_output)
            economic_parameters = EconomicParameters.from_builder(self, year)

            last_crop = _find_crop(self.crops, _last_planting(operations).crop) if operations else None

            result = self._find_best_rotation(year, doy, last_crop, economic_parameters)
            self._append_operations(result, year, doy, operations)
            self._times_planted[result.crop.symbol] += 1

            generate_operation_file(Path('./input') / f'{self.simulation}.operation', operations)


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
                rm_input=True,
                rm_output=True,
            )
            self.yield_matrix[crop.name] = self._read_yield_matrix(crop)


    def _find_best_rotation(self, year: int, doy: int, last_crop: Crop | None, economic_parameters: EconomicParameters) -> RotationResult:
        best = RotationResult(crop=self.crops[0], doy=0, n_rate=None, economic_return=float('-inf'))

        for crop1, crop2 in product(self.crops, self.crops):
            # Apply hard constraints: no soybean after soybean
            if last_crop and crop1.symbol == last_crop.symbol == 'GLMA4': continue
            if crop1.symbol == crop2.symbol == 'GLMA4': continue

            planting1 = _last_planting(crop1.operations)
            planting2 = _last_planting(crop2.operations)

            doys1 = np.arange(planting1.doy, planting1.end_doy + 1)
            doys2 = np.arange(planting2.doy, planting2.end_doy + 1)

            result = _calculate_economic_return(year, doy, doys1, doys2, last_crop, crop1, crop2, economic_parameters)

            if self.rotation_frequency is not None:
                rotation_year = year - self.control_dict['simulation_start_year'] + 1
                result = RotationResult(
                    crop=result.crop,
                    doy=result.doy,
                    n_rate=result.n_rate,
                    economic_return=result.economic_return + self._frequency_adjustment(rotation_year, crop1)
                )

            if result.economic_return > best.economic_return:
                best = result

        return best


    def _append_operations(self, result: RotationResult, year: int, doy: int, operations: list[Operation]) -> None:
        start_year = self.control_dict['simulation_start_year']
        planting_year = year - start_year + 1 if result.doy > doy else year - start_year + 2

        for op in result.crop.operations:
            relative_doy = False
            assert op.doy is not None
            if isinstance(op, Planting):
                operations.append(replace(op, year=planting_year, doy=result.doy))
            elif isinstance(op, (Tillage, FixedFertilization)):
                if op.doy < 0:
                    op_doy = result.doy + op.doy
                    op_year = planting_year
                else:
                    op_doy = op.doy
                    op_year = planting_year
                    relative_doy = True

                if isinstance(op, FixedFertilization) and result.n_rate is not None:
                    operations.append(replace(op, year=op_year, doy=op_doy, mass=result.n_rate / result.crop.prescribed_n_rate * op.mass if result.crop.prescribed_n_rate > 0 else 0.0, relative_doy=relative_doy))
                else:
                    operations.append(replace(op, year=op_year, doy=op_doy, relative_doy=relative_doy))
            else:
                operations.append(op)


    def _read_yield_matrix(self, crop: Crop) -> pd.DataFrame:
        df = pd.read_csv(f'summary/{crop.name}.csv', usecols=[1, 3, 6, 7, 18], comment='#')
        df['date'] = pd.to_datetime(df['date'])
        df['planting_date'] = pd.to_datetime(df['planting_date'])
        df['doy'] = pd.to_datetime(df['planting_date']).dt.dayofyear
        df['year'] = pd.to_datetime(df['planting_date']).dt.year
        df['growing_window'] = (df['date'] - df['planting_date']).dt.days
        df.drop(columns=['date', 'planting_date'], inplace=True)

        start_year = self.control_dict['simulation_start_year']
        end_year = self.control_dict['simulation_end_year']
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


    def _frequency_adjustment(self, year: int, crop: Crop) -> float:
        if year < 2:
            return 0.0

        assert self.rotation_frequency is not None
        frequency = self._times_planted[crop.symbol] / year
        lo, hi = self.rotation_frequency[crop.symbol]

        if frequency > hi:
            return -1e6
        elif frequency < lo:
            return 1e6 * (lo - frequency)
        return 0.0


def _calculate_economic_return(year: int, doy: int, doys1: np.ndarray, doys2: np.ndarray, last_crop: Crop | None, crop1: Crop, crop2: Crop, economic_parameters: EconomicParameters) -> RotationResult:
    penalty1 = economic_parameters.penalty_factors.get((last_crop.name, crop1.name), 0.0) if last_crop else 0.0
    penalty2 = economic_parameters.penalty_factors.get((crop1.name, crop2.name), 0.0)

    s1 = _sample_yield_matrix(economic_parameters.yield_matrix[crop1.name], doys1)
    s2 = _sample_yield_matrix(economic_parameters.yield_matrix[crop2.name], doys2)

    grain_yield1 = s1['grain_yield'][:, None]
    forage_yield1 = s1['forage_yield'][:, None]
    growing_window1 = s1['growing_window'][:, None]
    nitrogen_in_harvest1 = s1['nitrogen_in_harvest'][:, None]
    grain_yield2 = s2['grain_yield'][None, :]
    forage_yield2 = s2['forage_yield'][None, :]
    growing_window2 = s2['growing_window'][None, :]
    nitrogen_in_harvest2 = s2['nitrogen_in_harvest'][None, :]

    total_days = _calculate_total_window(doy, doys1[:, None], growing_window1, doys2[None, :], growing_window2)

    is_legume1 = _find_crop_group(crop1.symbol).name == 'DICOT_LEGUME'
    is_legume2 = _find_crop_group(crop2.symbol).name == 'DICOT_LEGUME'
    n_rate1 = 0.0 if is_legume1 else nitrogen_in_harvest1 * 1.33
    n_rate2 = 0.0 if is_legume2 else nitrogen_in_harvest2 * 1.33

    total_income = ((grain_yield1 + forage_yield1) * economic_parameters.crop_price[crop1.symbol] * (1.0 - penalty1) +
        (grain_yield2 + forage_yield2) * economic_parameters.crop_price[crop2.symbol] * (1.0 - penalty2))

    if economic_parameters.production_cost is not None:
        total_income -= economic_parameters.production_cost[crop1.symbol] + economic_parameters.production_cost[crop2.symbol]

    if economic_parameters.fertilizer_price is not None:
        fertilizer_cost1 = sum(n_rate1 / crop1.prescribed_n_rate * op.mass * economic_parameters.fertilizer_price[op.source] for op in crop1.operations if isinstance(op, FixedFertilization))
        fertilizer_cost2 = sum(n_rate2 / crop2.prescribed_n_rate * op.mass * economic_parameters.fertilizer_price[op.source] for op in crop2.operations if isinstance(op, FixedFertilization))
        total_income -= fertilizer_cost1 + fertilizer_cost2

    daily_incomes = total_income / total_days
    idx = np.unravel_index(np.argmax(daily_incomes), daily_incomes.shape)

    return RotationResult(
        crop=crop1,
        doy=int(doys1[idx[0]]),
        n_rate=None if is_legume1 else float(nitrogen_in_harvest1[idx[0], 0]),
        economic_return=float(daily_incomes[idx]),
    )


def _write_operation_templates(crops: list[Crop]) -> None:
    for crop in crops:
        path = Path('template') / f'{crop.name}.operation'
        path.parent.mkdir(exist_ok=True)
        lines: list[str] = []
        for ind, op in enumerate(crop.operations):
            overrides = {'doy': f'$DOY{ind + 1}'}
            if isinstance(op, Planting):
                overrides['end_doy'] = f'$DOY{ind + 1}'
            lines.extend(format_operation(op, overrides))
        path.write_text('\n'.join(lines))


def _build_simulations(operations: list[Operation], user_dict: dict) -> tuple[list[dict], dict, dict]:
    planting = _last_planting(operations)
    operation_dict: dict = {}
    simulations: list[dict] = []

    assert planting.doy is not None

    for ind, op in enumerate(operations):
        key = f'DOY{ind + 1}'
        operation_dict[key] = lambda x, k=key: x[k]

    for doy in range(planting.doy, planting.end_doy + 1):
        sim: dict = {'simulation_name': f'{planting.crop}_{doy}'}
        for ind, op in enumerate(operations):
            assert op.doy is not None
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


def _sample_yield_matrix(yield_df: pd.DataFrame, doys: np.ndarray) -> dict[str, np.ndarray]:
    sampled = yield_df.groupby('doy').sample(1).set_index('doy')
    return {col: sampled.loc[doys, col].to_numpy()
        for col in ('grain_yield', 'forage_yield', 'nitrogen_in_harvest', 'growing_window')}


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


def _find_crop_group(symbol: str) -> CropGroup:
    group = next((g for g in CROP_GROUPS if symbol in g.crops), None)
    if group is None:
        raise ValueError(f'Crop symbol {symbol} not found in any crop group.')
    return group


def _find_break_doy(output: str) -> tuple[int, int]:
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


def _calculate_total_window(doy: int, doys1: np.ndarray, window1: np.ndarray, doys2: np.ndarray, window2: np.ndarray) -> np.ndarray:
    temp_doy = (doys1 + window1) % 365
    return (doys1 - doy + 365 * (doy + MIN_PLANTING_INTERVAL > doys1) + window1
        + doys2 - temp_doy + 365 * (temp_doy + MIN_PLANTING_INTERVAL > doys2) + window2)


def _day_of_year(day: int) -> int:
    return day if day <= 365 else day - 365


def _optional_csv(path: Path | str | None) -> pd.DataFrame | None:
    return pd.read_csv(path, index_col=0) if path is not None else None


def _parse_value(line: str) -> tuple[str, float]:
    content = line.split('#')[0].split()
    return content[0].lower(), float(content[1])


def _read_fertilizer_file(path: str | Path) -> dict[str, Fertilizer]:
    fertilizers: dict[str, Fertilizer] = {}
    current_name: str | None = None
    current_data: dict[str, float] = {}

    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        tokens = line.split()
        if tokens[0].upper() == 'FERTILIZER':
            if current_name is not None:
                fertilizers[current_name] = Fertilizer(name=current_name, **current_data)
            current_name = tokens[1]
            current_data = {}
        else:
            key, value = _parse_value(line)
            current_data[key] = value

    # Flush the last block
    if current_name is not None:
        fertilizers[current_name] = Fertilizer(name=current_name, **current_data)

    return fertilizers
