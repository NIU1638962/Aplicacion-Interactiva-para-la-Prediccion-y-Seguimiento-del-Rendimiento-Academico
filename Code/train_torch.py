# -*- coding: utf-8 -*- noqa
"""
Created on Thu Mar 27 22:23:11 2025

@author: Joel Tapia Salvador
"""
from typing import Callable, Dict, Tuple, Union
import environment
import utils


def main_train(
        model: environment.torch.nn.Module,
        loss_function: environment.torch.nn.Module,
        optimizer: environment.torch.optim.Optimizer,
        number_epochs: int,
        train_dataloader: environment.torch.utils.data.DataLoader,
        validation_dataloader: Union[
            environment.torch.utils.data.DataLoader,
            None,
        ],
        added_metrics: Dict[
            str,
            Callable[
                [environment.torch.Tensor, environment.torch.Tensor],
                float,
            ],
        ] = {},
        objective: Tuple[str] = ('loss', 'minimize', 'validation'),
):
    if validation_dataloader is None:
        phases = ('train')
    else:
        phases = ('train', 'validation')

    time_log = {}

    metrics_log = {
        name_metric: {
            phase: [] for phase in phases
        } for name_metric in added_metrics.keys()
    }

    dataloaders = {}

    for phase in phases:
        metrics_log[phase]['loss'] = []
        time_log[phase] = []
        dataloaders[phase] = locals()[f'{phase}__dataloader']
        del locals()[f'{phase}__dataloader']

    best_model = BestModel(model.state_dict(), objective[1])

    for epoch in range(1, number_epochs + 1):
        epoch_information = f'Running epoch {epoch}/{number_epochs}'
        environment.logging.info(epoch_information.replace('\n', '\n\t\t'))
        print(environment.SEPARATOR_LINE + '\n' + epoch_information)

        for phase in phases:
            model, mean_loss, epoch_time = __epoch(
                model,
                loss_function,
                optimizer,
                dataloaders[phase],
                phase,
            )

            metrics_log['loss'][phase].append(mean_loss)
            time_log[phase].append(epoch_time)
            del mean_loss, epoch_time
            utils.collect_memory()

        del phase
        utils.collect_memory()

        best_model.update(
            epoch,
            time_log[objective[2]][epoch - 1],
            metrics_log[objective[0]][objective[2]],
            model.state_dict(),
        )


def __epoch(
    model: environment.torch.nn.Module,
    loss_function: environment.torch.nn.Module,
    optimizer: environment.torch.optim.Optimizer,
    dataloader: environment.torch.utils.data.DataLoader,
    phase: str,
    added_metrics: Dict[
        str,
        Callable[
            [environment.torch.Tensor, environment.torch.Tensor],
            float,
        ],
    ],
):
    if phase == 'train':
        model.train()
    elif phase == 'validation':
        model.eval()
    else:
        error = (
            f'Unexpected "{phase}" phase. Expected "train" or "validation".'
        )
        environment.logging.error(error.replace('\n', '\n\t\t'))
        raise AttributeError(error)

    with environment.torch.set_grad_enabled(phase == 'train'):
        running_epoch_loss = 0.0

        added_mretics_results = {
            name_metric: 0
            for name_metric in added_metrics.keys()
        }

        epoch_start_time = environment.time()

        for batch_index, (inputs, targets) in enumerate(dataloader):
            batch_size = inputs.size(0)
            optimizer.zero_grad()

            inputs = inputs.to(environment.TORCH_DEVICE)
            outputs = model(inputs)

            del inputs
            utils.collect_memory()

            targets = targets.to(environment.TORCH_DEVICE)
            step_loss = loss_function(outputs, targets)

            if loss_function.reduction == 'mean':
                running_epoch_loss += step_loss.item() * batch_size
            elif loss_function.reduction == 'sum':
                running_epoch_loss += step_loss.item()
            else:
                error = 'Only accepted loss reduction is "mean" or "sum".'
                environment.logging.error(error.replace('\n', '\n\t\t'))
                raise AttributeError(error)

            if phase == 'train':
                step_loss.backward()
                optimizer.step()

            del step_loss
            utils.collect_memory()

            for metric_name, function_metric in added_metrics:
                added_metrics[metric_name] += function_metric(outputs, targets)

            del outputs, targets
            utils.collect_memory()

        epoch_end_time = environment.time()
        epoch_time = epoch_end_time - epoch_start_time

        mean_epoch_loss = running_epoch_loss / len(dataloader.dataset)

    return model, mean_epoch_loss, epoch_time, added_mretics_results


class BestModel():

    __slots__ = (
        '__epoch_number',
        '__epoch_time',
        '__model_metric',
        '__model_parameters',
        '__objective',
    )

    def __init__(
        self,
        starting_parameters: environment.torch.nn.parameter.Parameter,
        objective: str = 'minimize',
    ):
        if objective in ('minimize', 'min', '-'):
            self.__objective = 'minimize'
            initial_metric = float('inf')
        elif objective in ('maximize', 'max', '+'):
            self.__objective = 'minimize'
            initial_metric = float('-inf')
        else:
            error = (
                'Objective is a not supported one.'
                + ' Supported objectives are "minimize" or "maximize".'
                + f' Got "{objective}" instead.'
            )
            environment.logging.error(error.replace('\n', '\n\t\t'))
            raise ValueError(error)

        self.__set(0, 0.0, initial_metric, starting_parameters)

    def __set(
        self,
        epoch_number: int,
        epoch_time: float,
        model_metric: float,
        model_parameters: environment.torch.nn.parameter.Parameter,
    ):
        if not isinstance(epoch_number, int):
            error = (
                f'"epoch_number" is not an integer, is a {type(epoch_number)}.'
            )
            environment.logging.error(error.replace('\n', '\n\t\t'))
            raise TypeError(error)

        if not isinstance(epoch_time, float):
            error = (
                f'"epoch_time" is not a float, is a {type(epoch_time)}.'
            )
            environment.logging.error(error.replace('\n', '\n\t\t'))
            raise TypeError(error)

        if not isinstance(model_metric, float):
            error = (
                f'"model_metric" is not a float, is a {type(epoch_number)}.'
            )
            environment.logging.error(error.replace('\n', '\n\t\t'))
            raise TypeError(error)

        if not isinstance(
                model_parameters,
                environment.torch.nn.parameter.Parameter,
        ):
            error = (
                '"model_parameters" is not a Torch Parameter'
                + f', is a {type(model_parameters)}.'
            )
            environment.logging.error(error.replace('\n', '\n\t\t'))
            raise TypeError(error)

        if epoch_number < 0:
            error = '"epoch_number" cannot be negative.'
            environment.logging.error(error.replace('\n', '\n\t\t'))
            raise ValueError(error)

        if epoch_time < 0:
            error = '"epoch_time" cannot be negative.'
            environment.logging.error(error.replace('\n', '\n\t\t'))
            raise ValueError(error)

        self.__epoch_number = epoch_number
        self.__epoch_time = epoch_time
        self.__model_metric = model_metric
        self.__model_parameters = environment.deepcopy(model_parameters)

    def update(
        self,
        new_epoch_number: int,
        new_epoch_time: float,
        new_model_metric: float,
        new_model_parameters: environment.torch.nn.parameter.Parameter,
    ) -> bool:
        """
        Update values of best model if the new metric is better than previous.

        Better is controled by the initialized "objective" parameter in the
        object's creation. If set to "minimize" a lesser value will be
        considered better. If set to "maximize" a greater value will be
        considered better.

        The function will deepcopy the parameters if it updates them,
        so there is no need to deepcopy them before passing them to the
        function.

        Parameters
        ----------
        new_epoch_number : Integer
            Number of the epoch where the new model was trained.
        new_epoch_time : Float
            Duration that the epoch took to train the new model.
        new_model_metric : Float
            Value of the comparasing metric valued over the new model.
        new_model_parameters : Torch Parameter
            Internal parameters of the new model.

        Raises
        ------
        ValueError
            Any of the funtions parameters has a not allowed value for what it
            represents or the objective has been set to a not recognized mode.

        Returns
        -------
        is_better : Boolean
            Wheter the new model is better than the previous best one.

        """
        if self.__objective == 'minimize':
            is_better = new_model_metric < self.__model_metric
        elif self.__objective == 'maximize':
            is_better = new_model_metric > self.__model_metric
        else:
            error = (
                'Objective is a not supported one.'
                + ' Supported objectives are "minimize" or "maximize".'
                + f' Got "{self.__objective}" instead.'
            )
            environment.logging.error(error.replace('\n', '\n\t\t'))
            raise ValueError(error)

        if is_better:
            self.__set(
                new_epoch_number,
                new_epoch_time,
                new_model_metric,
                new_model_parameters,
            )

        return is_better

    @property
    def epoch_number(self) -> int:
        """
        Getter for epoch number.

        Returns
        -------
        Integer
            Number of the epoch where the best model was trained.

        """
        return self.__epoch_number

    @property
    def epoch_time(self) -> float:
        """
        Getter for epoch time.

        Returns
        -------
        Float
            Duration that the epoch took to train the best model.

        """
        return self.__epoch_time

    @property
    def model_metric(self) -> float:
        """
        Getter for model metric.

        Returns
        -------
        Float
            Value of the comparasing metric valued over the best model.

        """
        return self.__model_metric

    @property
    def model_parameters(self) -> environment.torch.nn.parameter.Parameter:
        """
        Getter for model parameters.

        Returns
        -------
        Torch Parameter
            Internal parameters of the best model.

        """
        return self.__model_parameters
