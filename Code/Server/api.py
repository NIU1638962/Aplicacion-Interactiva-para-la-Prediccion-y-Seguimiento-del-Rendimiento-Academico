# -*- coding: utf-8 -*- noqa
"""
Created on Sat Jun 21 03:59:28 2025

@author: JoelT
"""
import datasets
import environment
import model
import utils


class ModelApi():
    __slots__ = (
        '__class_order',
        '__class_order_file_path',
        '__fields',
        '__fields_file_path',
        '__fields_keys',
        '__model',
        '__model_file_path',
        '__original_dataset_file_path',
        '__dataset_header',
    )

    def __init__(
            self,
            class_order_file_path: str,
            fields_file_path: str,
            model_file_path: str,
            original_dataset_file_path: str,
    ):
        self.__class_order_file_path = class_order_file_path
        self.__fields_file_path = fields_file_path
        self.__model_file_path = model_file_path
        self.__original_dataset_file_path = original_dataset_file_path

        self.api_reload()

    def api_get_fields(self):
        return self.__fields

    def api_input_validation(
            self,
            **parameters,
    ):
        response = {
            'valid': True,
            'errors': [],
        }

        parameters_keys = set(parameters.keys())

        for extra_parameter in parameters_keys - self.__fields_keys:
            response['valid'] = False
            response['errors'].append(
                f'"{extra_parameter}" is not a valid parameter.'
            )

        for key, field in self.__fields.items():
            if key not in parameters.keys():
                response['valid'] = False
                response['errors'].append(
                    f'"{key}" parameter is missing.'
                )
                continue

            if field['data_type'] == 'int':
                if not isinstance(parameters[key], (int)):
                    response['valid'] = False
                    response['errors'].append(
                        f'"{key}" parameter is not an integer number.'
                    )
                    continue

            elif field['data_type'] == 'float':
                if not isinstance(parameters[key], (int, float)):
                    response['valid'] = False
                    response['errors'].append(
                        f'"{key}" parameter is not a number.'
                    )
                    continue

            else:
                response['valid'] = False
                response['errors'].append(
                    f'"{key}" parameter is not expected data type.'
                )
                continue

            if field['type'] == 'select':
                if (
                    parameters[key] not in field['options'].values()
                ) and (
                    parameters[key] != -1
                ):
                    response['valid'] = False
                    response['errors'].append(
                        f'"{key}" value is not a valid option'
                        + f', recived "{parameters[key]}".'
                    )
                    continue

            elif field['type'] == 'range':
                if not (
                    field['range'][0] <= parameters[key] <= field['range'][1]
                ) and (
                    parameters[key] != -1
                ):
                    response['valid'] = False
                    response['errors'].append(
                        f'"{key}" value is not a valid option'
                        + f', recived "{parameters[key]}".'
                    )
                    continue

            else:
                response['valid'] = False
                response['errors'].append(
                    f'"{key}" parameter is not expected type.'
                )
                continue

        return response

    def api_predictor(
        self,
        **parameters,
    ):
        response = self.api_input_validation(**parameters)
        response['result'] = None

        if not response['valid']:
            return response

        dataset = self.to_dataset(**parameters)

        response['result'] = environment.torch.tensor(
            environment.numpy.clip(
                environment.numpy.round(self.__model.predict(dataset.inputs)),
                0,
                4,
            ).astype('int64'),
        ).item() * 2.5

        return response

    def api_recommendator(
        self,
        **parameters,
    ):
        response = self.api_input_validation(**parameters)
        response['result'] = None

        if not response['valid']:
            return response

        best_possible_class = max(
            self.__class_order.items(),
            key=lambda x: x[1],
        )[0]
        original_class = self.api_predictor(**parameters)['result']
        original_rank = self.__class_order[original_class]
        best_parameters = environment.deepcopy(parameters)
        best_class = original_class
        best_rank = original_rank
        best_changes = {
            feature_name: best_parameters[feature_name] != parameters[feature_name] for feature_name, field in self.__fields.items() if field['modifiable']
        }

        not_known_counter = 0
        modifiable_features_counter = 0

        if original_class != best_possible_class:
            for feature_name, field in self.__fields.items():
                if field['modifiable']:
                    modifiable_features_counter += 1
                    if parameters[feature_name] == -1:
                        not_known_counter += 1
                        continue

                    for value in self.__fields[feature_name]['possible_values']:
                        parameters_mod = environment.deepcopy(parameters)
                        parameters_mod[feature_name] = value

                        new_class = self.api_predictor(
                            **parameters_mod)['result']

                        new_rank = self.__class_order[new_class]

                        if new_rank > best_rank:
                            best_class = new_class
                            best_rank = new_rank
                            best_parameters = environment.deepcopy(
                                parameters_mod)
                            best_changes = {
                                feature_name: best_parameters[feature_name] != parameters[feature_name] for feature_name, field in self.__fields.items() if field['modifiable']
                            }

        response['result'] = {
            "original_class": original_class,
            "improved_class": best_class,
            "modified_parameters": best_parameters,
            "feature_changes": best_changes,
            "no_change_possible": not_known_counter == modifiable_features_counter,
            "best_class_already": original_class == best_possible_class,
        }

        return response

    def api_reload(self):
        response = {
            'reloaded': True,
            'errors': [],
        }
        self.__class_order = utils.load_json(self.__class_order_file_path)

        self.__fields = utils.load_json(self.__fields_file_path)

        for key in list(self.__class_order.keys()):
            if isinstance(key, str):
                try:
                    new_key = int(key)
                except ValueError:
                    try:
                        new_key = float(key)
                    except ValueError:
                        if key.lower() in ('none', 'null'):
                            new_key = None
                        else:
                            new_key = key

                if new_key != key:
                    self.__class_order[new_key] = self.__class_order[key]
                    del self.__class_order[key]

        self.__dataset_header = utils.load_csv(
            self.__original_dataset_file_path,
        ).columns.drop(['target']).to_list()

        if not environment.os.path.exists(self.__model_file_path):
            self.api_train_model()

        self.__model = utils.load_pickle(self.__model_file_path)

        self.__fields_keys = set(self.__fields.keys())

        for index, header in enumerate(self.__dataset_header):
            if header not in self.__fields.keys():
                raise ValueError(f'Missing field {header} in fields file.')

            if self.__fields[header]['type'] == 'select':
                self.__fields[header]['possible_values'] = [
                    value for value in self.__fields[header]['options'].values() if (
                        isinstance(value, (int, float))
                    ) and (
                        value != -1
                    )
                ]

            elif self.__fields[header]['type'] == 'range':
                if self.__fields[header]['data_type'] == 'int':
                    self.__fields[header]['possible_values'] = list(
                        range(
                            self.__fields[header]['range'][0],
                            self.__fields[header]['range'][-1] + 1,
                        ),
                    )

            if self.__fields[header]['data_type'] == 'float':
                self.__fields[header]['possible_values'] = [
                    self.__fields[header]['range'][0],
                ]

                position = environment.deepcopy(
                    self.__fields[header]['range'][0],
                )

                for iteration in range(
                    (
                        self.__fields[header]['range'][-1] -
                        self.__fields[header]['range'][0]
                    ),
                ):
                    for step in self.__fields[header]['float_steps']:
                        self.__fields[header]['possible_values'].append(
                            position + step,
                        )

                    position += 1

                    self.__fields[header]['possible_values'].append(position)

        return response

    def api_train_model(self):
        response = {
            'trained': True,
            'errors': [],
        }

        dataframe = utils.load_csv(
            self.__original_dataset_file_path,
        )

        dataset = datasets.Dataset(
            environment.torch.from_numpy(
                dataframe[self.__dataset_header].to_numpy(),
            ),
            environment.torch.from_numpy(
                dataframe['target'].astype('int64').to_numpy(),
            ),
            self.__dataset_header,
        )

        utils.save_pickle(
            model.MODEL(**model.PARAMETRES).fit(
                dataset.inputs,
                dataset.targets,
            ),
            self.__model_file_path,
        )

        return response

    def normalize_inputs(
        self,
        **parameters,
    ):
        for key, field in self.__fields.items():
            if key in parameters.keys():
                if field['type'] == 'range':
                    parameters[key] = utils.safe_division(
                        parameters[key] - field['mu'],
                        field['sigma'],
                        float('inf'),
                    )

        return parameters

    def to_dataset(
        self,
        **parameters,
    ):
        dataset = datasets.Dataset(
            environment.torch.from_numpy(
                environment.pandas.DataFrame(
                    data=[self.normalize_inputs(**parameters)],
                    columns=self.__dataset_header,
                )[self.__dataset_header].to_numpy()
            ),
            environment.torch.tensor([-1]),
            self.__dataset_header
        )

        return dataset
