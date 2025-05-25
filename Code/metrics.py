# -*- coding: utf-8 -*- noqa
"""
Created on Tue Apr  1 17:15:59 2025

@author: Joel Tapia Salvador
"""
from abc import ABC
from typing import List

import environment
import utils


def metric_method(func):
    """
    Decorator that tags a function as a metric method.
    Used to identify which methods should be registered in other classes
    containing the basic BinaryClassMetrics
    """
    func.is_metric = True
    return func


class MetricsClass(ABC):
    pass


class BinaryClassMetrics(MetricsClass):
    __slots__ = ("__tp", "__tn", "__fp", "__fn", '__n')

    @environment.torch.no_grad()
    def __init__(self):
        self.reset()

    @environment.torch.no_grad()
    def reset(self):
        """
        Reset internal attributes to calculate metric form batch 0 again.

        Returns
        -------
        None.

        """
        self.__tp = 0
        self.__tn = 0
        self.__fp = 0
        self.__fn = 0
        self.__n = 0

    @environment.torch.no_grad()
    def update(
            self,
            predicted: environment.torch.Tensor,
            targets: environment.torch.Tensor,
    ):
        """
        Update the internal representation with the new batch predictions.

        Parameters
        ----------
        outputs : environment.torch.Tensor
            DESCRIPTION.
        targets : environment.torch.Tensor
            DESCRIPTION.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        predicted = predicted.detach().int()
        targets = targets.detach().int()

        self.__tp += int(environment.torch.sum(
            (predicted == targets) & (predicted == 1)
        ))
        self.__tn += int(environment.torch.sum(
            (predicted == targets) & (predicted == 0)
        ))
        self.__fp += int(environment.torch.sum(
            (predicted != targets) & (predicted == 1)
        ))
        self.__fn += int(environment.torch.sum(
            (predicted != targets) & (predicted == 0)
        ))

        self.__n = self.__tp + self.__tn + self.__fp + self.__fn

    @metric_method
    @environment.torch.no_grad()
    def n(self) -> int:
        return self.__tp + self.__tn + self.__fp + self.__fn

    @metric_method
    @environment.torch.no_grad()
    def false_negative(self) -> int:
        return self.__fn

    @metric_method
    @environment.torch.no_grad()
    def false_positive(self) -> int:
        return self.__fp

    @metric_method
    @environment.torch.no_grad()
    def true_negative(self) -> int:
        return self.__tn

    @metric_method
    @environment.torch.no_grad()
    def true_positive(self) -> int:
        return self.__tp

    @metric_method
    @environment.torch.no_grad()
    def accuracy(self) -> float:
        return utils.safe_division(self.__tp + self.__tn, self.__n)

    @metric_method
    @environment.torch.no_grad()
    def balanced_accuracy(self) -> float:
        return utils.safe_division(
            self.true_positive_rate() + self.true_negative_rate(),
            2,
        )

    @metric_method
    @environment.torch.no_grad()
    def bookmaker_informedness(self) -> float:
        return self.informedness()

    @metric_method
    @environment.torch.no_grad()
    def critical_success_index(self) -> float:
        return self.threat_score()

    @metric_method
    @environment.torch.no_grad()
    def delta_p(self) -> float:
        return self.markedness()

    @metric_method
    @environment.torch.no_grad()
    def diagnostic_odds_ratio(self) -> float:
        return utils.safe_division(
            self.positive_likelihood_ratio(),
            self.negative_likelihood_ratio(),
        )

    @metric_method
    @environment.torch.no_grad()
    def fall_out(self) -> float:
        return self.false_discovery_rate()

    @metric_method
    @environment.torch.no_grad()
    def false_discovery_rate(self) -> float:
        return utils.safe_division(self.__fp, self.__tp + self.__fp)

    @metric_method
    @environment.torch.no_grad()
    def false_negative_rate(self) -> float:  # miss rate
        return utils.safe_division(self.__fn, self.__tp + self.__fn)

    @metric_method
    @environment.torch.no_grad()
    def false_omission_rate(self) -> float:
        return utils.safe_division(self.__fn, self.__tn + self.__fn)

    @metric_method
    @environment.torch.no_grad()
    def false_positive_rate(self) -> float:
        return utils.safe_division(self.__fp, self.__fp + self.__tn)

    @metric_method
    @environment.torch.no_grad()
    def f1_score(self) -> float:
        return utils.safe_division(
            2 * self.__tp,
            2 * self.__tp + self.__fp + self.__fn,
        )

    @metric_method
    @environment.torch.no_grad()
    def fowlkes_mallows_index(self) -> float:
        return (
            self.positive_predictive_value() * self.true_positive_rate()
        ) ** (
            1 / 2
        )

    @metric_method
    @environment.torch.no_grad()
    def hit_rate(self) -> float:
        return self.true_positive_rate()

    @metric_method
    @environment.torch.no_grad()
    def informedness(self) -> float:
        return self.true_positive_rate() + self.true_negative_rate() - 1

    @metric_method
    @environment.torch.no_grad()
    def jaccard_index(self) -> float:
        return self.threat_score()

    @metric_method
    @environment.torch.no_grad()
    def matthews_correlation_coefficient(self) -> float:
        return utils.safe_division(
            (
                self.__tp * self.__tn
            ) - (
                self.__fp * self.__fn
            ),
            (
                (
                    self.__tp + self.__fp
                ) * (
                    self.__tp + self.__fn
                ) * (
                    self.__tn + self.__fp
                ) * (
                    self.__tn + self.__fn
                )
            ) ** (
                1 / 2
            ),
        )

    @metric_method
    @environment.torch.no_grad()
    def markedness(self) -> float:
        return (
            self.positive_predictive_value() + self.negative_predictive_value() - 1
        )

    @metric_method
    @environment.torch.no_grad()
    def miss_rate(self) -> float:
        return self.false_negative_rate()

    @metric_method
    @environment.torch.no_grad()
    def negative_likelihood_ratio(self) -> float:
        return utils.safe_division(
            self.false_negative_rate(),
            self.true_negative_rate(),
        )

    @metric_method
    @environment.torch.no_grad()
    def negative_predictive_value(self) -> float:
        return utils.safe_division(self.__tn, self.__tn + self.__fn)

    @metric_method
    @environment.torch.no_grad()
    def positive_likelihood_ratio(self) -> float:
        return utils.safe_division(
            self.true_positive_rate(),
            self.false_positive_rate(),
        )

    @metric_method
    @environment.torch.no_grad()
    def positive_predictive_value(self) -> float:
        return utils.safe_division(self.__tp, self.__tp + self.__fp)

    @metric_method
    @environment.torch.no_grad()
    def precision(self) -> float:
        return self.positive_predictive_value()

    @metric_method
    @environment.torch.no_grad()
    def prevalence(self) -> float:
        return utils.safe_division(self.__tp + self.__fn, self.__n)

    @metric_method
    @environment.torch.no_grad()
    def prevalence_threshold(self) -> float:
        tpr = self.true_positive_rate()
        fpr = self.false_positive_rate()
        return utils.safe_division(
            (tpr * fpr) ** (1 / 2) - fpr,
            tpr - fpr,
        )

    @metric_method
    @environment.torch.no_grad()
    def probability_of_detection(self) -> float:
        return self.true_positive_rate()

    @metric_method
    @environment.torch.no_grad()
    def probability_of_false_alarm(self) -> float:
        return self.false_positive_rate()

    @metric_method
    @environment.torch.no_grad()
    def recall(self) -> float:
        return self.true_positive_rate()

    @metric_method
    @environment.torch.no_grad()
    def selectivity(self) -> float:
        return self.true_negative_rate()

    @metric_method
    @environment.torch.no_grad()
    def sensitivity(self) -> float:
        return self.true_positive_rate()

    @metric_method
    @environment.torch.no_grad()
    def specificity(self) -> float:
        return self.true_negative_rate()

    @metric_method
    @environment.torch.no_grad()
    def threat_score(self) -> float:
        return utils.safe_division(
            self.__tp,
            self.__tp + self.__fn + self.__fp,
        )

    @metric_method
    @environment.torch.no_grad()
    def true_negative_rate(self) -> float:
        return utils.safe_division(self.__tn, self.__fp + self.__tn)

    @metric_method
    @environment.torch.no_grad()
    def true_positive_rate(self) -> float:   # recall / sensitivity
        return utils.safe_division(self.__tp, self.__tp + self.__fn)

    # def __repr__(self) -> str:
    #     m = ", ".join(f"{k}={v:.4f}" for k, v in self.summary().items())
    #     return f"{self.__class__.__name__}({m})"


class MultiLabelMetrics(MetricsClass):
    __slots__ = ('__metrics', '__number_classes', '__weights')

    def __init__(self, number_classes: int, weights: environment.torch.Tensor):
        if weights.shape[0] != number_classes:
            raise ValueError(
                "positive_weight must have the same length as number_classes")

        self.__number_classes = number_classes

        self.__metrics = [
            BinaryClassMetrics() for _ in range(self.__number_classes)
        ]

        self.__weights = weights.to(environment.TORCH_DEVICE)

        self.__register_metrics()

    @environment.torch.no_grad()
    def reset(self):
        """
        Reset all metrics and true label counts.

        Returns
        -------
        None.

        """
        for metric in self.__metrics:
            metric.reset()

    @environment.torch.no_grad()
    def update(
            self,
            predicted: environment.torch.Tensor,
            targets: environment.torch.Tensor,
    ):
        """
        Update the internal representation with the new batch predictions.

        Parameters
        ----------
        predicted : environment.torch.Tensor
            DESCRIPTION.
        targets : environment.torch.Tensor
            DESCRIPTION.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        predicted = predicted.int()
        targets = targets.int()

        for i in range(self.__number_classes):
            # Update metrics for each class individually
            self.__metrics[i].update(predicted[:, i], targets[:, i])

    @environment.torch.no_grad()
    def __weighted(self, values: List[float]) -> float:
        """
        Compute a weighted average of metric values using class label counts.

        Parameters
        ----------
        values : List[float]
            DESCRIPTION.

        Returns
        -------
        float
            DESCRIPTION.

        """
        total_weight = self.__weights.sum().item()
        if total_weight == 0:
            return 0.0

        values_tensor = environment.torch.tensor(
            values,
            dtype=environment.torch.float32,
            device=environment.TORCH_DEVICE,
        )

        return float((values_tensor * self.__weights).sum() / total_weight)

    @environment.torch.no_grad()
    def __register_metrics(self):
        """
        Dynamically register metrics from BinaryClassMetrics.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        for name_atrribute in BinaryClassMetrics.__dict__:
            attribute = BinaryClassMetrics().__getattribute__(name_atrribute)

            # Only register methods tagged with @metric_method
            if callable(attribute) and getattr(attribute, "is_metric", False):
                def make_metric_method(name):
                    def method(self):
                        # Compute the weighted average for each class's metric
                        values = [getattr(m, name)() for m in self.__metrics]
                        return self.__weighted(values)
                    method.__name__ = name
                    method.is_metric = True
                    return method

                setattr(
                    self.__class__,
                    name_atrribute,
                    make_metric_method(name_atrribute),
                )

    # def __repr__(self) -> str:
    #     """
    #     String representation of the current metrics, formatted neatly.

    #     Returns
    #     -------
    #     str
    #         DESCRIPTION.

    #     """
    #     m = ", ".join(f"{k}={v:.4f}" for k, v in self.summary().items())
    #     return f"{self.__class__.__name__}({m})"


class RegressionMetrics(MetricsClass):

    @environment.torch.no_grad()
    def __init__(self):
        self.predictions = environment.torch.empty(
            0,
            device=environment.TORCH_DEVICE,
        )
        self.targets = environment.torch.empty(
            0,
            device=environment.TORCH_DEVICE,
        )

    @environment.torch.no_grad()
    def update(
        self,
        predictions: environment.torch.Tensor,
        targets: environment.torch.Tensor,
    ):
        """Append a batch of predictions and targets to the storage."""
        predictions = predictions.view(-1).to(environment.TORCH_DEVICE)
        targets = targets.view(-1).to(environment.TORCH_DEVICE)
        self.predictions = environment.torch.cat(
            [self.predictions, predictions],
            dim=0,
        )
        self.targets = environment.torch.cat([self.targets, targets], dim=0)

    @environment.torch.no_grad()
    def reset(self):
        self.predictions = environment.torch.empty(
            0,
            device=environment.TORCH_DEVICE,
        )
        self.targets = environment.torch.empty(
            0,
            device=environment.TORCH_DEVICE,
        )

    @metric_method
    @environment.torch.no_grad()
    def mae(self):
        return environment.torch.mean(
            environment.torch.abs(self.predictions - self.targets)
        ).item()

    @metric_method
    @environment.torch.no_grad()
    def mse(self):
        return environment.torch.mean(
            (self.predictions - self.targets) ** 2
        ).item()

    @metric_method
    @environment.torch.no_grad()
    def rmse(self):
        return environment.torch.sqrt(
            environment.torch.mean((self.predictions - self.targets) ** 2)
        ).item()

    @metric_method
    @environment.torch.no_grad()
    def r2(self):
        ss_res = environment.torch.sum(
            (self.targets - self.predictions) ** 2
        )
        ss_tot = environment.torch.sum(
            (self.targets - environment.torch.mean(self.targets)) ** 2
        )
        return (1 - ss_res / ss_tot).item()

    # @metric_method
    # @environment.torch.no_grad()
    # def adjusted_r2(self, num_features: int):
    #     n = self.targets.numel()
    #     return 1 - (1 - self.r2()) * (n - 1) / (n - num_features - 1)

    # @metric_method
    # @environment.torch.no_grad()
    # def aic(self, num_features: int):
    #     n = self.targets.numel()
    #     return (
    #         n * environment.torch.log(self.mse()) + 2 * (num_features + 1)
    #     ).item()

    # @metric_method
    # @environment.torch.no_grad()
    # def bic(self, num_features: int):
    #     n = self.targets.numel()
    #     return (
    #         n * environment.torch.log(self.mse()) + (num_features + 1) * environment.torch.log(
    #             environment.torch.tensor(
    #                 n, dtype=environment.torch.float, device=environment.TORCH_DEVICE)
    #         )
    #     ).item()

    # @metric_method
    # @environment.torch.no_grad()
    # def f_statistic(self, num_features: int):
    #     n = self.targets.numel()
    #     k = num_features
    #     # regression sum of squares
    #     ssr = environment.torch.sum(
    #         (self.predictions - environment.torch.mean(self.targets)) ** 2
    #     )
    #     # residual sum of squares
    #     sse = environment.torch.sum((self.targets - self.predictions) ** 2)
    #     msr = ssr / k
    #     mse = sse / (n - k - 1)
    #     return (msr / mse).item()


def test_batch_metrics_function_equality(
    metric: str,
    number_batches: int = 10,
    range_batches=(2, 11),
    seed=None,
):
    environment.numpy.random.seed(seed)
    batches = {}

    data = {
        'outputs': environment.torch.tensor([]),
        'targets': environment.torch.tensor([]),
    }

    binary_class_metrics = BinaryClassMetrics()

    for i in range(number_batches):
        batches[f'batch_{i}'] = {}
        len_batch = environment.numpy.random.randint(*range_batches)

        batches[f'batch_{i}']['outputs'] = environment.torch.from_numpy(
            environment.numpy.random.randint(0, 2, len_batch)
        )
        batches[f'batch_{i}']['targets'] = environment.torch.from_numpy(
            environment.numpy.random.randint(0, 2, len_batch)
        )

        data['outputs'] = environment.torch.cat(
            (data['outputs'], batches[f'batch_{i}']['outputs'])
        )
        data['targets'] = environment.torch.cat(
            (data['targets'], batches[f'batch_{i}']['targets'])
        )

        binary_class_metrics.update(
            batches[f'batch_{i}']['outputs'],
            batches[f'batch_{i}']['targets'],
        )

        batch_metric = binary_class_metrics.__getattribute__(metric)()

        binary_class_metrics.reset()

        binary_class_metrics.update(
            data['outputs'],
            data['targets'],
        )

        real_metric = binary_class_metrics.__getattribute__(metric)()

        equal = batch_metric == real_metric

        print(f'Metric calculating by batch: {batch_metric}')
        print(f'Real metric: {real_metric}')
        print(f'Are they equal: {equal}')


AVAILABLE_METRICS_CLASSES = {
    'binary_class_metrics': BinaryClassMetrics,
    'multi_label_metrics': MultiLabelMetrics,
}


if __name__ == '__main__':
    for name_atrribute in BinaryClassMetrics.__dict__:
        attribute = BinaryClassMetrics().__getattribute__(name_atrribute)
        if getattr(attribute, "is_metric", False):
            print(environment.SEPARATOR_LINE)
            print(f'{name_atrribute}:')
            test_batch_metrics_function_equality(name_atrribute, seed=0)
