# -*- coding: utf-8 -*- noqa
"""
Created on Thu Mar 27 01:35:33 2025

@author: Joel Tapia Salvador
"""
from typing import Tuple, Union

import environment

from available_torch_losses import AVAILABLE_LOSSES
from available_torch_optimizers import AVAILABLE_OPTIMIZERS
from parameters import get_number_of_model_parameters, get_parameters_to_update, initialize_weights


def init_model(
    model_type: environment.torch.nn.Module,
    model_init_parameters: dict,
    loss_type: Union[environment.torch.nn.Module, str],
    loss_init_parameters: dict,
    optimizer_type: Union[environment.torch.optim.Optimizer, str],
    optimizer_init_parameters: dict,
    weights_init_type: str,
    bench_mark: bool = False,
    manual_seed: Union[int, None] = None,
) -> Tuple[
    environment.torch.nn.Module,
    environment.torch.nn.Module,
    environment.torch.optim.Optimizer,
]:
    """
    Initialize model, weights, loss and optimizer for the training.

    Parameters
    ----------
    model_type : Torch Module
        Model type.
    model_init_parameters : Dictionary
        Parameters to initilaize the model.
    loss_type : Torch Module or String
        Loss type.
    loss_init_parameters : Dictionary
        Paramters to initialize the loss.
    optimizer_type : Torch Optimizer or String
        Type of optimizer to use.
    optimizer_init_parameters : Dictionary
        Parameters to initialize the optimizer. Does not include the model
        internal parameters, those are obtained by the funtion.
    weights_init_type : String
        Type of weight initialization used. Options:
            - 'xavier_normal'
            - 'xavier_uniform'
            - 'kaiming_normal'
            - 'kaiming_uniform'
            - 'orthogonal'
    bench_mark : Boolean, optional
        Wether activate CUDA benchmark. The default is False.
    manual_seed : Integer or None, optional
        Seed for reproductibity. The default is None.

    Returns
    -------
    model : Torch Module
        Model to be trained.
    loss : Torch Module
        Loss function for the training.
    optimizer : Torch Optimizer
        Optimizer for the training.

    """
    # Activate CUDA benchamrks or not (slows down training if activated)
    environment.torch.backends.cudnn.benchmark = bench_mark

    # Seed for reproductibility
    if manual_seed is not None:
        environment.torch.manual_seed(manual_seed)

    # Initialize model
    model = model_type(
        **model_init_parameters
    )
    number_of_parameters = get_number_of_model_parameters(model)
    verbose_number_of_parameters = (
        f'Number of parameters: {number_of_parameters}'
    )
    environment.logging.info(
        verbose_number_of_parameters.replace('\n', '\n\t\t')
    )
    print(verbose_number_of_parameters)

    # Initialize weight of the model
    initialize_weights(model, weights_init_type)

    # Move model to device
    model.to(environment.TORCH_DEVICE)

    # Inicialize loss
    loss = AVAILABLE_LOSSES.get(loss_type, loss_type)(
        **loss_init_parameters,
    )

    # Get parameters to optimize and update while training
    (
        name_parameters_optimize,
        parameters_optimize,
        number_of_parameters_optimize
    ) = get_parameters_to_update(model)
    verbose_number_of_parameters_optimize = (
        f'Number of parameters: {number_of_parameters_optimize}'
    )
    environment.logging.info(
        verbose_number_of_parameters_optimize.replace('\n', '\n\t\t')
    )
    print(verbose_number_of_parameters_optimize)
    verbose_name_parameters_optimize = (
        'Parameters that will be updated during training:\n\t- '
        + f'{"chr(10)chr(9)- ".join(name_parameters_optimize)}'
    )
    environment.logging.info(
        verbose_name_parameters_optimize.replace('\n', '\n\t\t')
    )
    print(verbose_name_parameters_optimize)

    # Inicialize optimizer
    optimizer = AVAILABLE_OPTIMIZERS.get(optimizer_type, optimizer_type)(
        parameters_optimize,
        **optimizer_init_parameters,
    )

    return model, loss, optimizer
