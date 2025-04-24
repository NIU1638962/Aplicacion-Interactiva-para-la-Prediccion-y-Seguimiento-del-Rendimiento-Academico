# -*- coding: utf-8 -*- noqa
"""
Created on Tue Apr  1 17:15:59 2025

@author: Joel Tapia Salvador
"""
import environment


def test_batch_metrics_function_equality(
        metric_function,
        number_batches: int = 10,
        range_batches=(2, 11),
        seed=None,
):
    environment.numpy.random.seed(seed)
    batches = {}
    batches_metric = {}

    batches_metric_total = 0

    data = {
        'outputs': environment.torch.tensor([]),
        'targets': environment.torch.tensor([]),
    }

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

        batches_metric[f'batch_{i}'] = metric_function(
            outputs=batches[f'batch_{i}']['outputs'],
            targets=batches[f'batch_{i}']['targets'],
        )

        batches_metric_total += (
            batches_metric[f'batch_{i}'] *
            batches[f'batch_{i}']['targets'].size()[0]
        )

    mean_batch_metric = batches_metric_total / data['targets'].size()[0]
    real_metric = metric_function(
        outputs=data['outputs'],
        targets=data['targets'],
    )

    print(f'Metric calculating by batch: {mean_batch_metric}')
    print(f'Real metric: {real_metric}')


def __keywords(**kwargs):
    arguments_name = sorted(kwargs.keys())
    if len(kwargs) == 2:
        if not (
            (
                'outputs' in arguments_name
            ) and (
                'targets' in arguments_name
            )
        ):
            error = (
                'Expected ("targets", "outputs") arguments, got: (' +
                ", ".join(arguments_name) + ').'
            )
            environment.logging.error(error.replace('\n', '\n\t\t'))
            raise TypeError(error)

        return compare(kwargs['outputs'], kwargs['targets'])

    if len(kwargs) == 4:
        if not (
            (
                'true_positive' in arguments_name
            ) and (
                'true_negative' in arguments_name
            ) and (
                'false_positive' in arguments_name
            ) and (
                'false_negative' in arguments_name
            )
        ):
            error = (
                'Expected ("false_negative", "false_positive", ' +
                '"true_negative", "true_positive") arguments, got: (' +
                ", ".join(arguments_name) + ').'
            )
            environment.logging.error(error.replace('\n', '\n\t\t'))
            raise TypeError(error)

        return (
            kwargs['true_positive'],
            kwargs['true_negative'],
            kwargs['false_positive'],
            kwargs['false_negative'],
        )

    error = (
        f'Expected 2 or 4 arguments, got {len(kwargs)} instead.'
    )
    environment.logging.error(error.replace('\n', '\n\t\t'))
    raise TypeError(error)


def compare(outputs, targets):
    true_positive = environment.torch.sum(
        (outputs == targets) & (outputs == 1)
    )
    true_negative = environment.torch.sum(
        (outputs == targets) & (outputs == 0)
    )
    false_positive = environment.torch.sum(
        (outputs != targets) & (outputs == 1)
    )
    false_negative = environment.torch.sum(
        (outputs != targets) & (outputs == 0)
    )

    return true_positive, true_negative, false_positive, false_negative


def accuracy(**kwargs):
    true_positive, true_negative, false_positive, false_negative = __keywords(
        **kwargs,
    )

    return (
        true_positive + true_negative
    ) / (
        true_positive + true_negative + false_positive + false_negative
    )


def balanced_accuracy(**kwargs):
    return (
        true_positive_rate(**kwargs) +
        true_negative_rate(**kwargs)
    ) / (
        2
    )


def bookmaker_informedness(**kwargs):
    return informedness(**kwargs)


def critical_success_index(**kwargs):
    return threat_score(**kwargs)


def diagnostic_oddss_ratio(**kwargs):
    return (
        positive_likelihood_ratio(**kwargs)
    ) / (
        negative_likelihood_ratio(**kwargs)
    )


def delta_p(**kwargs):
    return markedness(**kwargs)


def fall_out(**kwargs):
    return false_positive_rate(**kwargs)


def false_discovery_rate(**kwargs):
    true_positive, true_negative, false_positive, false_negative = __keywords(
        **kwargs,
    )
    return (
        false_positive
    ) / (
        true_positive + false_positive
    )


def false_negative_rate(**kwargs):
    true_positive, true_negative, false_positive, false_negative = __keywords(
        **kwargs,
    )
    return (
        false_negative
    ) / (
        true_positive + false_negative
    )


def false_omission_rate(**kwargs):
    true_positive, true_negative, false_positive, false_negative = __keywords(
        **kwargs,
    )
    return (
        false_negative
    ) / (
        true_negative + false_negative
    )


def false_positive_rate(**kwargs):
    true_positive, true_negative, false_positive, false_negative = __keywords(
        **kwargs,
    )
    return (
        false_positive
    ) / (
        false_positive + true_negative
    )


def f1_score(**kwargs):
    true_positive, true_negative, false_positive, false_negative = __keywords(
        **kwargs,
    )
    return (
        2 * true_positive
    ) / (
        2 * true_positive + false_positive + false_negative
    )


def fowlkes_mallows_index(**kwargs):
    return (
        positive_predictive_value(**kwargs) *
        true_positive_rate(**kwargs)
    ) ** (
        1 / 2
    )


def hit_rate(**kwargs):
    return true_positive_rate(**kwargs)


def informedness(**kwargs):
    return (
        true_positive_rate(**kwargs) +
        true_negative_rate(**kwargs) - 1
    )


def jaccard_index(**kwargs):
    return threat_score(**kwargs)


def matthews_correlation_coefficient(**kwargs):
    true_positive, true_negative, false_positive, false_negative = __keywords(
        **kwargs,
    )
    return (
        (
            true_positive * true_negative
        ) - (
            false_positive * false_negative
        )
    ) / (
        (
            (
                true_positive + false_positive
            ) * (
                true_positive + false_negative
            ) * (
                true_negative + false_positive
            ) * (
                true_negative + false_negative
            )
        ) ** (1/2)
    )


def markedness(**kwargs):
    return (
        positive_predictive_value(**kwargs) +
        negative_predictive_value(**kwargs) - 1
    )


def miss_rate(**kwargs):
    return false_negative_rate(**kwargs)


def negative_likelihood_ratio(**kwargs):
    return (
        false_negative_rate(**kwargs) /
        true_negative_rate(**kwargs)
    )


def negative_predictive_value(**kwargs):
    true_positive, true_negative, false_positive, false_negative = __keywords(
        **kwargs,
    )
    return (
        true_negative
    ) / (
        true_negative + false_negative
    )


def positive_likelihood_ratio(**kwargs):
    return (
        true_positive_rate(**kwargs) /
        false_positive_rate(**kwargs)
    )


def positive_predictive_value(**kwargs):
    true_positive, true_negative, false_positive, false_negative = __keywords(
        **kwargs,
    )
    return (
        true_positive
    ) / (
        true_positive + false_positive
    )


def precision(**kwargs):
    return positive_predictive_value(**kwargs)


def prevalence(**kwargs):
    true_positive, true_negative, false_positive, false_negative = __keywords(
        **kwargs,
    )
    return (
        true_positive + false_negative
    ) / (
        true_positive + false_negative + false_positive + true_negative
    )


def prevalence_threshold(**kwargs):
    tpr = true_positive_rate(**kwargs)
    fpr = false_positive_rate(**kwargs)
    return (
        (tpr * fpr) ** (1/2) - fpr
    ) / (
        tpr - fpr
    )


def probability_of_detection(**kwargs):
    return true_positive_rate(**kwargs)


def probability_of_false_alarm(**kwargs):
    return false_positive_rate(**kwargs)


def recall(**kwargs):
    return true_positive_rate(**kwargs)


def selectivity(**kwargs):
    return true_negative_rate(**kwargs)


def sensitivity(**kwargs):
    return true_positive_rate(**kwargs)


def specificity(**kwargs):
    return true_negative_rate(**kwargs)


def threat_score(**kwargs):
    true_positive, true_negative, false_positive, false_negative = __keywords(
        **kwargs,
    )
    return (
        true_positive
    ) / (
        true_positive + false_negative + false_positive
    )


def true_negative_rate(**kwargs):
    true_positive, true_negative, false_positive, false_negative = __keywords(
        **kwargs,
    )
    return (
        true_negative
    ) / (
        false_positive + true_negative
    )


def true_positive_rate(**kwargs):
    true_positive, true_negative, false_positive, false_negative = __keywords(
        **kwargs,
    )
    return (
        true_positive
    ) / (
        true_positive + false_negative
    )


if __name__ == '__main__':
    print('Accuracy:')
    test_batch_metrics_function_equality(accuracy, seed=0)
    print('Balanced Accuracy')
    test_batch_metrics_function_equality(balanced_accuracy, seed=0)
    print('Bookmaker Informedness')
    test_batch_metrics_function_equality(bookmaker_informedness, seed=0)
    print('Critical Success Index')
    test_batch_metrics_function_equality(critical_success_index, seed=0)
    print('Diagnostic Oddss Ratio')
    test_batch_metrics_function_equality(diagnostic_oddss_ratio, seed=0)
    print('Delta P')
    test_batch_metrics_function_equality(delta_p, seed=0)
    print('Fall Out')
    test_batch_metrics_function_equality(fall_out, seed=0)
    print('False Discovery Rate')
    test_batch_metrics_function_equality(false_discovery_rate, seed=0)
    print('False Negative Rate')
    test_batch_metrics_function_equality(false_negative_rate, seed=0)
    print('False Omission Rate')
    test_batch_metrics_function_equality(false_omission_rate, seed=0)
    print('False Positive Rate')
    test_batch_metrics_function_equality(false_positive_rate, seed=0)
    print('F1 Score')
    test_batch_metrics_function_equality(f1_score, seed=0)
    print('Fowlkes Mallows Index')
    test_batch_metrics_function_equality(fowlkes_mallows_index, seed=0)
    print('Hit Rate')
    test_batch_metrics_function_equality(hit_rate, seed=0)
    print('Informedness')
    test_batch_metrics_function_equality(informedness, seed=0)
    print('Jaccard Index')
    test_batch_metrics_function_equality(jaccard_index, seed=0)
    print('Matthews Correlation Coefficient')
    test_batch_metrics_function_equality(
        matthews_correlation_coefficient,
        seed=0,
    )
    print('Markedness')
    test_batch_metrics_function_equality(markedness, seed=0)
    print('Miss rate')
    test_batch_metrics_function_equality(miss_rate, seed=0)
    print('Negative Likelihood Ratio')
    test_batch_metrics_function_equality(negative_likelihood_ratio, seed=0)
    print('Negative Predictive Value')
    test_batch_metrics_function_equality(negative_predictive_value, seed=0)
    print('Positive Likelihood Ratio')
    test_batch_metrics_function_equality(positive_likelihood_ratio, seed=0)
    print('Positive Predictive Value')
    test_batch_metrics_function_equality(positive_predictive_value, seed=0)
    print('Precision')
    test_batch_metrics_function_equality(precision, seed=0)
    print('Prevalence')
    test_batch_metrics_function_equality(prevalence, seed=0)
    print('Prevalence Threshold')
    test_batch_metrics_function_equality(prevalence_threshold, seed=0)
    print('Probability Of Detection')
    test_batch_metrics_function_equality(probability_of_detection, seed=0)
    print('Probability Of False Alarm')
    test_batch_metrics_function_equality(probability_of_false_alarm, seed=0)
    print('Recall')
    test_batch_metrics_function_equality(recall, seed=0)
    print('Selectivity')
    test_batch_metrics_function_equality(selectivity, seed=0)
    print('Sensitivity')
    test_batch_metrics_function_equality(sensitivity, seed=0)
    print('Specificity')
    test_batch_metrics_function_equality(specificity, seed=0)
    print('Threat Score')
    test_batch_metrics_function_equality(threat_score, seed=0)
    print('True Negative Rate')
    test_batch_metrics_function_equality(true_negative_rate, seed=0)
    print('True Positive Rate')
    test_batch_metrics_function_equality(true_positive_rate, seed=0)
