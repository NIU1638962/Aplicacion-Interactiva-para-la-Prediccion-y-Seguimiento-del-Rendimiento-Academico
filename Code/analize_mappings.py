# -*- coding: utf-8 -*- noqa
"""
Created on Sun Jun 15 16:14:10 2025

@author: JoelT
"""
from datasets import get_datasets
import environment
import utils


def analyze_mapping(
    mapping,
    float_range_sample,
):
    result = {}

    for param, values in mapping.items():
        param_info = {}

        param_info['not_known'] = values.pop('not known', None)

        possible_values = sorted(list(values.values()))

        if len(possible_values) == 0:
            param_info['type'] = 'empty'
            param_info['min'] = None
            param_info['max'] = None
            param_info['possible_values'] = []

        elif (
                len(possible_values) == 1
        ) and (
            isinstance(possible_values[0], list)
        ):
            param_info['min'] = min(possible_values[0])
            param_info['max'] = max(possible_values[0])

            if param_info['min'] == 0 and param_info['max'] == 1:
                param_info['type'] = 'float_range'
                param_info['possible_values'] = environment.deepcopy(
                    float_range_sample,
                )

            else:
                param_info['type'] = 'integer_range'
                param_info['possible_values'] = list(
                    range(param_info['min'], param_info['max'] + 1),
                )

        else:
            param_info['type'] = 'categorical'
            param_info['possible_values'] = [
                possible_value for possible_value in possible_values if isinstance(
                    possible_value,
                    (int, float),
                )
            ]

            param_info['min'] = param_info['possible_values'][0]
            param_info['max'] = param_info['possible_values'][-1]

        result[param] = param_info

    return dict(sorted(result.items()))


if __name__ == '__main__':
    mapping = utils.load_json(
        environment.os.path.join(
            environment.DATA_PATH, 'general_mapping.json'),
    )

    # Analyze the mapping
    param_summary = analyze_mapping(
        mapping,
        [0, 0.25, 0.5, 0.75, 1],
    )

    # Print the results
    for param, info in param_summary.items():
        print(f"\nParameter: {param}")
        print(f"  Type           : {info['type']}")
        print(f"  Not known val  : {info['not_known']}")
        print(f"  Min            : {info['min']}")
        print(f"  Max            : {info['max']}")
        print(f"  Possible values: {info['possible_values']}")
