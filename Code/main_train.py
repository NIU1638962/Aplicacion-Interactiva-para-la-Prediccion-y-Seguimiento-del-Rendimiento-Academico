# -*- coding: utf-8 -*- noqa
"""
Created on Sun Apr 20 21:04:02 2025

@author: Joel Tapia Salvador
"""
import environment
import utils
import models_sklearn
import train_sklearn

from data_merge import merge_data
from datasets import get_datasets, k_fold
from metrics import MultiLabelMetrics, RegressionMetrics
from recomendator import test_recomendator


def main():
    utils.print_message(
        environment.SECTION_LINE
        + '\nSTARTING MAIN PROGRAM'

    )
    environment.NUMBER_GRADES = 5

    merge_data()

    full_dataset = get_datasets('full')

    environment.NUMBER_FOLDS = 5

    utils.print_message(
        environment.SECTION_LINE
        + '\nSTARTING TRAINING'

    )

    environment.EXPERIEMNT = 'experiment_1'

    dataset = get_datasets(environment.EXPERIEMNT)

    utils.plot_target_distributions(
        target_tensors=[full_dataset.targets, dataset.targets],
        number_targets=environment.NUMBER_GRADES,
        dataset_labels=['Full Dataset',  'Experiment 1 Dataset'],
        path=environment.os.path.join(
            environment.RESULTS_PATH,
            'full_vs_experiment_one_targets_distribution.png'
        ),
    )

    temp_metrics = MultiLabelMetrics(1, environment.torch.tensor([1]))

    metrics = [
        name_atrribute for name_atrribute in MultiLabelMetrics.__dict__ if getattr(
            temp_metrics.__getattribute__(name_atrribute),
            "is_metric",
            False,
        )
    ]

    temp_metrics = RegressionMetrics()

    metrics.extend(
        [
            name_atrribute for name_atrribute in RegressionMetrics.__dict__ if getattr(
                temp_metrics.__getattribute__(name_atrribute),
                "is_metric",
                False,
            )
        ]
    )

    del temp_metrics

    temp_results = environment.pandas.DataFrame(columns=metrics)

    temp_results.index.name = 'model'

    results = {
        'mean_fold': environment.deepcopy(temp_results),
        'std_fold': environment.deepcopy(temp_results),
    }

    look_up_names = environment.pandas.DataFrame(columns=['name'])
    look_up_names.index.name = 'hash_name'

    del temp_results
    utils.collect_memory()

    for model_type, model in models_sklearn.models.items():
        for parameters in models_sklearn.parameters[model_type]:
            name = (
                model_type
                + '--'
                + '--'.join(
                    [
                        f'{parameter_name}_{parameter_value}' for (
                            parameter_name,
                            parameter_value,
                        ) in sorted(parameters.items())
                    ]
                )
            )

            utils.print_message(
                f'{environment.SEPARATOR_LINE}\nTraining Model: {name}'
            )

            hash_name = utils.get_hash_string('md5', name)

            look_up_names.loc[hash_name] = [name]

            name = hash_name
            del hash_name

            temp_results = environment.pandas.DataFrame(columns=metrics)

            temp_results.index.name = 'fold'

            for fold, train_dataset, validation_dataset in k_fold(dataset, 5):
                utils.print_message(
                    f'{environment.SECTION * 3}'
                    + f'Fold: {fold}'
                    + f'{environment.SECTION * 3}'
                )

                current_results = train_sklearn.main_train(
                    model(**parameters),
                    train_dataset,
                    validation_dataset,
                )

                temp_results.loc[fold] = current_results

            utils.save_csv(
                temp_results,
                environment.os.path.join(
                    environment.RESULTS_PATH,
                    f'{environment.EXPERIEMNT}_{name}_folds_metrics.csv',
                ),
                index=True,
            )

            results['mean_fold'].loc[name] = temp_results.mean()
            results['std_fold'].loc[name] = temp_results.std()

    utils.save_csv(
        results['mean_fold'],
        environment.os.path.join(
            environment.RESULTS_PATH,
            f'{environment.EXPERIEMNT}_metrics_mean.csv',
        ),
        index=True,
    )

    utils.save_csv(
        results['std_fold'],
        environment.os.path.join(
            environment.RESULTS_PATH,
            f'{environment.EXPERIEMNT}_metrics_std.csv',
        ),
        index=True,
    )

    utils.save_csv(
        look_up_names,
        environment.os.path.join(
            environment.RESULTS_PATH,
            f'{environment.EXPERIEMNT}_look_up_names.csv',
        ),
        index=True,
    )

    environment.EXPERIEMNT = 'experiment_2'

    dataset = get_datasets(environment.EXPERIEMNT)

    temp_metrics = MultiLabelMetrics(1, environment.torch.tensor([1]))

    metrics = [
        name_atrribute for name_atrribute in MultiLabelMetrics.__dict__ if getattr(
            temp_metrics.__getattribute__(name_atrribute),
            "is_metric",
            False,
        )
    ]

    temp_metrics = RegressionMetrics()

    metrics.extend(
        [
            name_atrribute for name_atrribute in RegressionMetrics.__dict__ if getattr(
                temp_metrics.__getattribute__(name_atrribute),
                "is_metric",
                False,
            )
        ]
    )

    del temp_metrics

    temp_results = environment.pandas.DataFrame(columns=metrics)

    temp_results.index.name = 'model'

    results = {
        'mean_fold': environment.deepcopy(temp_results),
        'std_fold': environment.deepcopy(temp_results),
    }

    del temp_results
    utils.collect_memory()

    model_type = 'random_forest_classifier'

    model = models_sklearn.models[model_type]

    parameters = models_sklearn.parameters[model_type][2]

    name = (
        model_type
        + '--'
        + '--'.join(
            [
                f'{parameter_name}_{parameter_value}' for (
                    parameter_name,
                    parameter_value,
                ) in sorted(parameters.items())
            ]
        )
    )

    utils.print_message(
        f'{environment.SEPARATOR_LINE}\nTraining Model: {name}'
    )

    hash_name = utils.get_hash_string('md5', name)

    name = hash_name
    del hash_name

    temp_results = environment.pandas.DataFrame(columns=metrics)

    temp_results.index.name = 'fold'

    for fold, train_dataset, validation_dataset in k_fold(dataset, 5):
        utils.print_message(
            f'{environment.SECTION * 3}'
            + f'Fold: {fold}'
            + f'{environment.SECTION * 3}'
        )

        current_results = train_sklearn.main_train(
            model(**parameters),
            train_dataset,
            validation_dataset,
        )

        temp_results.loc[fold] = current_results

    utils.save_csv(
        temp_results,
        environment.os.path.join(
            environment.RESULTS_PATH,
            f'{environment.EXPERIEMNT}_{name}_folds_metrics.csv',
        ),
        index=True,
    )

    results['mean_fold'].loc[name] = temp_results.mean()
    results['std_fold'].loc[name] = temp_results.std()

    utils.save_csv(
        results['mean_fold'],
        environment.os.path.join(
            environment.RESULTS_PATH,
            f'{environment.EXPERIEMNT}_metrics_mean.csv',
        ),
        index=True,
    )

    utils.save_csv(
        results['std_fold'],
        environment.os.path.join(
            environment.RESULTS_PATH,
            f'{environment.EXPERIEMNT}_metrics_std.csv',
        ),
        index=True,
    )

    environment.EXPERIEMNT = 'experiment_3'

    dataset = get_datasets(environment.EXPERIEMNT)

    model = models_sklearn.models[model_type]

    parameters = models_sklearn.parameters[model_type][2]

    model = model(**parameters)

    model.fit(dataset.inputs, dataset.targets)

    raw_mapping = utils.load_json(
        environment.os.path.join(
            environment.DATA_PATH,
            'general_mapping.json',
        ),
    )

    class_order = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

    float_range_sample = [0, 0.25, 0.5, 0.75, 1]

    modifiable_feature_names = [
        'absences',
        'additional_work',
        'alcohol_weekend',
        'alcohol_workday',
        'attendance_classes',
        'attendance_seminars',
        # 'attendance_time',
        'extra_curricular_activities',
        'free_time',
        'go_out',
        'listening_classes',
        'personal_classes',
        'reading_frequency_non_scientific',
        'reading_frequency_scientific',
        'taking_notes_classes',
        'weekly_study_hours',
    ]

    results = test_recomendator(
        raw_mapping,
        float_range_sample,
        dataset,
        modifiable_feature_names,
        model,
        class_order,
    )


if __name__ == "__main__":
    main()
