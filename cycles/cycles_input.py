import numpy as np
import pandas as pd

SOIL_PARAMETERS = ['clay', 'sand', 'soc', 'bulk_density']
SOIL_LAYERS = [
    # units: top (m), bottom (m), thickness (m), NO3 (kg/ha), NH4 (Kg/ha)
    {'top': 0, 'bottom': 0.05, 'thickness': 0.05, 'no3': 10, 'nh4': 1},
    {'top': 0.05, 'bottom': 0.1, 'thickness': 0.05, 'no3': 10, 'nh4': 1},
    {'top': 0.1, 'bottom': 0.2, 'thickness': 0.1, 'no3': 7, 'nh4': 1},
    {'top': 0.2, 'bottom': 0.4, 'thickness': 0.2, 'no3': 4, 'nh4': 1},
    {'top': 0.4, 'bottom': 0.6, 'thickness': 0.2, 'no3': 2, 'nh4': 1},
    {'top': 0.6, 'bottom': 0.8, 'thickness': 0.2, 'no3': 1, 'nh4': 1},
    {'top': 0.8, 'bottom': 1.0, 'thickness': 0.2, 'no3': 1, 'nh4': 1},
    {'top': 1.0, 'bottom': 1.2, 'thickness': 0.2, 'no3': 1, 'nh4': 1},
    {'top': 1.2, 'bottom': 1.4, 'thickness': 0.2, 'no3': 1, 'nh4': 1},
    {'top': 1.4, 'bottom': 1.6, 'thickness': 0.2, 'no3': 1, 'nh4': 1},
    {'top': 1.6, 'bottom': 1.8, 'thickness': 0.2, 'no3': 1, 'nh4': 1},
    {'top': 1.8, 'bottom': 2.0, 'thickness': 0.2, 'no3': 1, 'nh4': 1},
]
CURVE_NUMBERS = {
    #row crops, SR, Good
    'A': 67,
    'B': 78,
    'C': 85,
    'D': 89,
}
CONTROL_PARAMETERS = {
    'simulation years': {
        'simulation_start_date': None,
        'simulation_end_date': None,
        'rotation_size': None,
    },
    'other input files': {
        'crop_file': 'GenericCrops.crop',
        'operation_file': None,
        'soil_file': None,
        'weather_file': None,
        'reinit_file': 'N/A',
    },
    'simulation options': {
        'co2_level': -999,
        'use_reinitialization': 0,
        'adjusted_yields': 0,
        'hourly_infiltration': 1,
        'automatic_nitrogen': 0,
        'automatic_phosphorus': 0,
        'automatic_sulfur': 0,
    },
    'output control': {
        'daily_weather_out': 1,
        'daily_crop_out': 1,
        'daily_residue_out': 1,
        'daily_water_out': 1,
        'daily_nitrogen_out': 1,
        'daily_soil_carbon_out': 1,
        'daily_soil_lyr_cn_out': 1,
        'annual_soil_out': 1,
        'annual_profile_out': 1,
        'annual_nflux_out': 1,
    }
}


def overlapping_depth(top1, bottom1, top2, bottom2):
    return max(0.0, min(bottom1, bottom2) - max(top1, top2))


def calculate_parameter(soil_df, parameter, top, bottom):
    soil_df['weight'] = soil_df.apply(lambda x: overlapping_depth(x['top'], x['bottom'], top, bottom) / (bottom - top), axis=1)
    soil_df = soil_df[soil_df['weight'] > 0]

    return np.sum(np.array(soil_df[parameter] * soil_df['weight'])) / sum(soil_df['weight'])


def generate_soil_file(fn, desc, hsg, slope, soil_df, soil_depth=None):
    layer_depths = np.array([layer['bottom'] for layer in SOIL_LAYERS])

    if soil_depth is not None:
        layer_depths = layer_depths[layer_depths <= soil_depth]

    soil_depth = min(layer_depths, key=lambda x: abs(x - soil_df.iloc[-1]['bottom']))

    df = pd.DataFrame.from_dict(
        {v: [layer[v] for layer in SOIL_LAYERS if layer['bottom'] <= soil_depth] for v in SOIL_LAYERS[0]}
    )

    df['layer'] = range(1, len(df) + 1)

    for v in SOIL_PARAMETERS:
        df[v] = df.apply(lambda x: calculate_parameter(soil_df, v, x['top'], x['bottom']), axis=1)

    cn = -999 if not hsg else CURVE_NUMBERS[hsg[0]]

    with open(fn, 'w') as f:
        f.write(desc)

        f.write("%-15s\t%d\n" % ("CURVE_NUMBER", cn))
        f.write("%-15s\t%.2f\n" % ("SLOPE", slope))

        f.write("%-15s\t%d\n" % ("TOTAL_LAYERS", len(df)))
        f.write(('%-7s\t'*12 + '%s\n') % (
            "LAYER", "THICK", "CLAY", "SAND", "SOC", "BD", "FC", "PWP", "SON", "NO3", "NH4", "BYP_H", "BYP_V"
        ))

        f.write(('%-7s\t'*12 + '%s\n') % (
            "#", "m", "%", "%", "%", "Mg/m3", "m3/m3", "m3/m3", "kg/ha", "kg/ha", "kg/ha", "-", "-"
        ))

        for _, row in df.iterrows():
            f.write('%-7d\t' % row['layer'])
            f.write('%-7.2f\t' % float(row['thickness']))
            f.write('%-7s\t' % '-999' if np.isnan(row['clay']) else '%-7.1f\t' % float(row['clay']))
            f.write('%-7s\t' % '-999' if np.isnan(row['sand']) else '%-7.1f\t' % float(row['sand']))
            f.write('%-7s\t' % '-999' if np.isnan(row['soc']) else '%-7.2f\t' % float(row['soc']))
            f.write('%-7s\t' % '-999' if np.isnan(row['bulk_density']) else '%-7.2f\t' % float(row['bulk_density']))
            f.write(('%-7d\t'*3 + '%-7.1f\t'*2 + '%-7.1f\t%.1f\n') % (
                -999, -999, -999, float(row['no3']), float(row['nh4']), 0.0, 0.0
            ))


def generate_control_file(fn, user_control_dict):
    with open(fn, 'w') as f:
        for block, parameters in CONTROL_PARAMETERS.items():
            f.write(f'## {block.upper()} ##\n')
            for name, value in parameters.items():
                if value is None:
                    try:
                        value = user_control_dict[name]
                    except:
                        raise KeyError(f'Parameter {name,upper()} must be defined')

                f.write('%-23s\t%s\n' % (name.upper(), str(value)))
            f.write('\n')
