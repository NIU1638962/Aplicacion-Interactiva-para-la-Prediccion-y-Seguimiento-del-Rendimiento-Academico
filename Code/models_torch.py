# -*- coding: utf-8 -*- noqa
"""
Created on Wed Apr 30 05:25:22 2025

@author: Joel Tapia Salvador
"""
import environment
import utils


class Predictor():

    def __init__(
            self,
            input_dimensions,
            hidden_layers=[],
            outputs_dimensions=[2, 1],
            threshold_method='roc_closest',
            dropout_rate=0.5,

    ):

        super().__init__()

        self.outputs_dimensions = outputs_dimensions

        self.sigmoid = environment.torch.nn.Sigmoid()

        self.sigmoid.should_initialize = False

        self.threshold_method = threshold_method

        self.thresholds = environment.torch.nn.ParameterList(
            [
                environment.torch.nn.Parameter(
                    environment.torch.full((num_classes,), 0.5),
                    requires_grad=False,
                ) for num_classes in self.number_classes]
        )

        layers = []
        input_dimension = input_dimensions[1]

        for hidden_dimension in hidden_layers:
            layers.append(environment.torch.nn.Dropout(p=dropout_rate))
            layers.append(environment.torch.nn.Linear(
                input_dimension,
                hidden_dimension,
            ))
            layers.append(environment.torch.nn.ReLU())
            input_dimension = hidden_dimension

        layers.append(environment.torch.nn.Dropout(p=dropout_rate))
        layers.append(environment.torch.nn.Linear(
            input_dimension,
            self.number_classes,
        ))

        self.classifier = environment.torch.nn.Sequential(*layers)

        for module in self.classifier.modules():
            module.should_initialize = True

    def forward(self, x):
        z = self.classifier(x)

        return [z]

    @environment.torch.no_grad()
    def results(self, list_logits):
        results = []
        for logits, threshold in zip(list_logits, self.thresholds):
            results.append(
                (
                    self.sigmoid(logits) >= threshold
                ).to(environment.torch.float32)
            )
        return results

    @environment.torch.no_grad()
    def update_threshold(self, list_logits, list_labels):
        if not self.training:
            return

        utils.print_message('Updating thresholds...')

        for (
            logits,
            labels,
            thresholds,
            number_classes,
        ) in zip(
            list_logits,
            list_labels,
            self.thresholds,
            self.number_classes,
        ):

            probabilities = self.sigmoid(logits)

            for c in range(number_classes):
                if self.threshold_method == 'pr':
                    thresholds.data[c] = self._update_threshold_pr(
                        probabilities[:, c],
                        labels[:, c],
                    ).item()
                elif self.threshold_method == 'roc_youden':
                    thresholds.data[c] = self._update_threshold_roc_youden(
                        probabilities[:, c],
                        labels[:, c],
                    ).item()
                elif self.threshold_method == 'roc_closest':
                    thresholds.data[c] = self._update_threshold_roc_closest(
                        probabilities[:, c],
                        labels[:, c],
                    ).item()

        utils.print_message('Thresholds updated.')

    @environment.torch.no_grad()
    def _update_threshold_roc_closest(self, probabilities, labels):
        fpr, tpr, thresholds = environment.sklearn.metrics.roc_curve(
            labels.cpu(),
            probabilities.cpu(),
        )

        return environment.torch.tensor(
            thresholds[
                environment.numpy.argmin(
                    environment.numpy.sqrt((1 - tpr) ** 2 + fpr ** 2)
                )
            ],
            dtype=environment.torch.float32,
            device=environment.TORCH_DEVICE,
        )


AVAILABLE_MODELS = {
    'predictor': Predictor,
}
