# -*- coding: utf-8 -*- noqa
"""
Created on Sun Apr 20 23:11:45 2025

@author: Joel Tapia Salvador
"""
import environment
import utils

from metrics import MultiLabelMetrics, RegressionMetrics


def main_train(
        model,
        train_dataset,
        validation_dataset,
):
    results = {}

    model.fit(train_dataset.inputs, train_dataset.targets)

    predictions = environment.torch.tensor(
        environment.numpy.clip(
            environment.numpy.round(model.predict(validation_dataset.inputs)),
            0,
            environment.NUMBER_GRADES - 1,
        ).astype('int64'),
    )

    metrics = RegressionMetrics()

    metrics.update(predictions, validation_dataset.targets)

    for name_atrribute in RegressionMetrics.__dict__:
        attribute = metrics.__getattribute__(name_atrribute)
        if getattr(attribute, "is_metric", False):
            calculated_metric = attribute()
            utils.print_message(f'{name_atrribute}: {calculated_metric}')
            results[name_atrribute] = environment.deepcopy(
                calculated_metric,
            )

    predictions_one_hot = utils.one_hot_encode(
        predictions,
        environment.NUMBER_GRADES,
    ).to(environment.TORCH_DEVICE)

    targets_one_hot = utils.one_hot_encode(
        validation_dataset.targets,
        environment.NUMBER_GRADES,
    ).to(environment.TORCH_DEVICE)

    metrics = MultiLabelMetrics(
        environment.NUMBER_GRADES,
        environment.torch.ones([environment.NUMBER_GRADES]),
    )

    metrics.update(predictions_one_hot, targets_one_hot)

    for name_atrribute in MultiLabelMetrics.__dict__:
        attribute = metrics.__getattribute__(name_atrribute)
        if getattr(attribute, "is_metric", False):
            calculated_metric = attribute()
            utils.print_message(f'{name_atrribute}: {calculated_metric}')
            results[name_atrribute] = environment.deepcopy(
                calculated_metric,
            )

    return results
