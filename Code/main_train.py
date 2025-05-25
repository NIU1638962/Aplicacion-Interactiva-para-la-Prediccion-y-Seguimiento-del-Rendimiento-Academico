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

    del temp_results
    utils.collect_memory()

    for name, model in models_sklearn.models.items():
        utils.print_message(
            f'{environment.SEPARATOR_LINE}\nTraining Model: {name}'
        )

        temp_results = environment.pandas.DataFrame(columns=metrics)

        temp_results.index.name = 'fold'

        for fold, train_dataset, validation_dataset in k_fold(dataset, 5):
            utils.print_message(
                f'{environment.SECTION * 3}'
                + f'Fold: {fold}'
                + f'{environment.SECTION * 3}'
            )

            current_results = train_sklearn.main_train(
                model(),
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

    name = 'random_forest'

    model = models_sklearn.models[name]

    utils.print_message(
        f'{environment.SEPARATOR_LINE}\nTraining Model: {name}'
    )

    temp_results = environment.pandas.DataFrame(columns=metrics)

    temp_results.index.name = 'fold'

    for fold, train_dataset, validation_dataset in k_fold(dataset, 5):
        utils.print_message(
            f'{environment.SECTION * 3}'
            + f'Fold: {fold}'
            + f'{environment.SECTION * 3}'
        )

        current_results = train_sklearn.main_train(
            model(),
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


if __name__ == "__main__":
    main()
