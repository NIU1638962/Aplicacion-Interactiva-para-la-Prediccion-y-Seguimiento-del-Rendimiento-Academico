# -*- coding: utf-8 -*- noqa
"""
Created on Tue Mar 25 21:29:37 2025

@author: Joel Tapia Salvador
"""
from typing import Tuple, Union

import environment


def collect_memory():
    """
    Collect garbage's collector's' memory and CUDA's cache and shared memory.

    Returns
    -------
    None.

    """
    environment.gc.collect()

    if environment.CUDA_AVAILABLE and environment.TORCH_DEVICE.type == 'cuda':
        environment.torch.cuda.ipc_collect()
        environment.torch.cuda.empty_cache()


def get_disk_space(path: str) -> Tuple[int, int, int]:
    """
    Get information about the path's disk's space.

    Print the disk's space information (total disk space, free disk space,
    used disk space) of the give path in human redable text and return such
    information in bytes.

    Parameters
    ----------
    path : String
        Path to look up disk's space.

    Returns
    -------
    total_disk_space : Integer
        Number of bytes of the total disk's space of the path.
    used_disk_space : Integer
        Number of bytes currently used of the path.
    free_disk_space : Integer
        Number of bytes currently free of the path.

    """
    disk_space = environment.psutil.disk_usage(path)

    total_disk_space = disk_space.total
    free_disk_space = disk_space.free

    used_disk_space = total_disk_space - free_disk_space

    redable_free_disk_space = transform_redable_byte_scale(free_disk_space)
    redable_total_disk_space = transform_redable_byte_scale(total_disk_space)
    redable_used_disk_space = transform_redable_byte_scale(used_disk_space)

    verbose_redable_disk_space_info = (
        f'Total Path Disk Space: {redable_total_disk_space}'
        + f'\nUsed Path Disk Space: {redable_used_disk_space}'
        + f'\nFree Path Disk Space: {redable_free_disk_space}'
    )

    environment.logging.info(
        verbose_redable_disk_space_info.replace('\n', '\n\t\t'))

    print_message(verbose_redable_disk_space_info)

    return total_disk_space, used_disk_space, free_disk_space


def get_hash_string(hash_type: str, string: str) -> str:
    hash_object = environment.hashlib.new(hash_type)

    hash_object.update(string.encode())

    hash_string = hash_object.hexdigest()

    return hash_string


def get_memory_cuda() -> Tuple[int, int, int]:
    """
    Get information about CUDA's device memory.

    Print the memory information (total memory, free memory, used memory) of
    the CUDA device in human redable text and return such information in bytes.

    Returns
    -------
    total_memory : Integer
        Number of bytes of the total memory of the CUDA device.
    used_memory : Integer
        Number of bytes currently used of the CUDA device.
    free_memory : Integer
        Number of bytes currently free of the CUDA device.

    """
    total_memory = 0
    free_memory = 0

    if environment.CUDA_AVAILABLE and environment.TORCH_DEVICE.type == 'cuda':
        free_memory, total_memory = environment.torch.cuda.mem_get_info(
            environment.TORCH_DEVICE
        )

    used_memory = total_memory - free_memory

    redable_free_memory = transform_redable_byte_scale(free_memory)
    redable_total_memory = transform_redable_byte_scale(total_memory)
    redable_used_memory = transform_redable_byte_scale(used_memory)

    verbose_redable_memory_info = (
        f'Total CUDA Memory: {redable_total_memory}'
        + f'\nUsed CUDA Memory: {redable_used_memory}'
        + f'\nFree CUDA Memory: {redable_free_memory}'
    )

    environment.logging.info(
        verbose_redable_memory_info.replace('\n', '\n\t\t'),
    )

    print_message(verbose_redable_memory_info)

    return total_memory, used_memory, free_memory


def get_memory_object(an_object: object) -> int:
    """
    Get object size in bytes.

    Warning: this does not include size of referenced objects inside the
    objejct and is teh result of calling a method of the object that can be
    overwritten. Be careful when using and interpreting results.

    Parameters
    ----------
    an_object : Object
        Object to get the size of.

    Returns
    -------
    size : Integer
        Size in bytes of the object.

    """
    size = environment.sys.getsizeof(an_object)

    return size


def get_memory_system() -> Tuple[int, int, int]:
    """
    Get information about system's memory.

    Print the memory information (total memory, free memory, used memory) of
    the system in human redable text and return such information in bytes.

    Returns
    -------
    total_memory : Integer
        Number of bytes of the total memory of the system.
    used_memory : Integer
        Number of bytes currently used of the system.
    free_memory : Integer
        Number of bytes currently free of the system.

    """
    memory = environment.psutil.virtual_memory()

    total_memory = memory.total
    free_memory = memory.available

    used_memory = total_memory - free_memory

    redable_free_memory = transform_redable_byte_scale(free_memory)
    redable_total_memory = transform_redable_byte_scale(total_memory)
    redable_used_memory = transform_redable_byte_scale(used_memory)

    verbose_redable_memory_info = (
        f'Total System Memory: {redable_total_memory}'
        + f'\nUsed System Memory: {redable_used_memory}'
        + f'\nFree System Memory: {redable_free_memory}'
    )

    environment.logging.info(
        verbose_redable_memory_info.replace('\n', '\n\t\t'),
    )

    print_message(verbose_redable_memory_info)

    return total_memory, used_memory, free_memory


def load_csv(
    file_path: str,
    encoding: str = environment.locale.getpreferredencoding(),
    header='infer',
    skip_blank_lines: bool = True,
    low_memory: bool = True,
    index_column=None,
) -> environment.pandas.DataFrame:
    """
    Load and read a CSV file into a Pandas DataFrame.

    Parameters
    ----------
    file_path : String
        String with the relative or absolute path of the CSV file.
    encoding : String, optional
        String with the encoding used on the CSV file. The default is
        locale.getpreferredencoding().

    Returns
    -------
    data : Pandas DataFrame
        Pandas Dataframe containing the CSV data.

    """
    with open(file_path, mode='r', encoding=encoding) as csv_file:
        dialect = environment.csv.Sniffer().sniff(csv_file.readline())

    del csv_file
    collect_memory()

    data = environment.pandas.read_csv(
        file_path,
        sep=dialect.delimiter,
        header=header,
        skip_blank_lines=skip_blank_lines,
        low_memory=low_memory,
        index_col=index_column,
    )

    del dialect
    collect_memory()

    return data


def load_json(
    file_path: str,
    encoding: str = environment.locale.getpreferredencoding(),
) -> object:
    """
    Load a JSON file into a Python Object.

    Parameters
    ----------
    file_path : String
        String with the relative or absolute path of the JSON file.
    encoding : String, optional
        String with the encoding used on the JSON file. The default is
        locale.getpreferredencoding().

    Returns
    -------
    python_object : Object
        Python Object containing the JSON data.

    """
    with open(file_path, mode='r', encoding=encoding) as file:
        python_object = environment.json.load(file)

    del file
    collect_memory()

    return python_object


def load_pickle(file_path: str) -> object:
    """
    Read a binary dump made by pickle of a Python Object.

    Parameters
    ----------
    file_path : Sting
        String with the relative or absolute path of the binary file.

    Returns
    -------
    python_object : Object
        Python Object read from the binary dump made by pickle.

    """
    with open(file_path, mode='rb') as file:
        python_object = environment.pickle.load(file)

    del file
    collect_memory()

    return python_object


def module_from_file(
    module_name: str,
    file_path: str,
) -> environment.types.ModuleType:
    """
    Load a Python ModuleType from a file.

    Parameters
    ----------
    module_name : String
        Name of the module.
    file_path : String
        Path to the module file.

    Returns
    -------
    module : Python ModuleType
        Python ModuleType.

    """
    spec = environment.importlib.util.spec_from_file_location(
        module_name, file_path
    )
    module = environment.importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def one_hot_encode(targets, number_targets):
    """
    number_targets

    Parameters
    ----------
    targets : TYPE
        1D tensor of class indices with shape [x].
    number_targets : TYPE
        Tensor: One-hot encoded tensor of shape [x, N].

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return environment.torch.nn.functional.one_hot(
        targets,
        num_classes=number_targets,
    ).float()


def plot_feature_importance(
    fit_model,
    feature_names,
    path,
    title=None,
    highlight_features=None,
    highlight_color="tab:red",
    default_color="tab:blue",
    highlight_label="Important Feature",
    default_label="Other Feature",

):
    if hasattr(fit_model, "feature_importances_"):
        importances = fit_model.feature_importances_
        label = "Feature Importance"
    elif hasattr(fit_model, "coef_"):
        importances = fit_model.coef_
        if len(importances.shape) > 1:
            importances = environment.numpy.mean(importances, axis=0)
        label = "Coefficient"
    else:
        raise ValueError("Model does not have feature_importances_ or coef_.")

    # Sort by absolute magnitude
    indices = environment.numpy.argsort(
        environment.numpy.abs(importances),
    )[::-1]
    sorted_importances = importances[indices]
    sorted_names = [feature_names[i] for i in indices]

    # Determine bar colors
    highlight_features = set(
        highlight_features
    ) if highlight_features else set()
    bar_colors = [
        highlight_color if name in highlight_features else default_color
        for name in sorted_names
    ]

    # Plot
    environment.matplotlib.pyplot.figure(figsize=(10, 6))
    environment.matplotlib.pyplot.title(title or "Feature Weights")
    bars = environment.matplotlib.pyplot.bar(
        range(len(sorted_importances)),
        sorted_importances,
        color=bar_colors,
    )
    environment.matplotlib.pyplot.xticks(
        range(len(sorted_importances)),
        sorted_names,
        rotation=90,
    )
    environment.matplotlib.pyplot.ylabel(label)
    environment.matplotlib.pyplot.tight_layout()

    if highlight_features:
        if any(name in highlight_features for name in sorted_names):
            legend_patches = [
                environment.matplotlib.patches.Patch(
                    color=highlight_color,
                    label=highlight_label,
                ),
                environment.matplotlib.patches.Patch(
                    color=default_color,
                    label=default_label,
                ),
            ]
            environment.matplotlib.pyplot.legend(
                handles=legend_patches,
                loc="best",
            )

    environment.matplotlib.pyplot.savefig(path)


def plot_target_distributions(
    target_tensors,
    number_targets,
    target_labels=None,
    dataset_labels=None,
    path='',
):
    """


    Parameters
    ----------
    target_tensors : TYPE
        List of M tensors, contianing targets index.
    number_targets : TYPE
        Number of targets.
    target_labels : TYPE, optional
        Labels for the number_targets. Defaults to range(number_targets).
    dataset_labels : TYPE, optional
        Labels for the M datasets. Defaults to "Dataset 1", "Dataset 2", etc.
        The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Convert to numpy for plotting
    data = []
    for tensor in target_tensors:
        if isinstance(tensor, environment.torch.Tensor):
            tensor = tensor.cpu().numpy()
        histogram = environment.numpy.bincount(
            tensor,
            minlength=number_targets,
        )
        data.append(histogram)

    number_datasets = len(data)

    # Sanity check
    for array in data:
        if len(array) != number_targets:
            raise ValueError(
                'All target tensors must have the same number of targets.'
            )

    # Default labels
    if target_labels is None:
        target_labels = [f'Class {index}' for index in range(number_targets)]
    if dataset_labels is None:
        dataset_labels = [
            f'Dataset {index + 1}' for index in range(number_datasets)
        ]

    x = environment.numpy.arange(number_targets)
    width = 0.8 / number_datasets

    fig, ax = environment.matplotlib.pyplot.subplots(figsize=(10, 6))

    for index, array in enumerate(data):
        ax.bar(x + index * width, array, width, label=dataset_labels[index])

    ax.set_xlabel('Target Classes')
    ax.set_ylabel('Counts')
    ax.set_title('Target Distributions Across Datasets')
    ax.set_xticks(x + width * (number_datasets - 1) / 2)
    ax.set_xticklabels(target_labels)
    ax.legend()
    environment.matplotlib.pyplot.grid(
        True,
        axis='y',
        linestyle='--',
        alpha=0.5,
    )
    environment.matplotlib.pyplot.tight_layout()
    environment.matplotlib.pyplot.savefig(path)


def print_message(message: str):
    """
    Print a message to sys.stdout and log it and "info" level.

    Parameters
    ----------
    message : String
        Message to print.

    Returns
    -------
    None.

    """
    environment.logging.info(message.replace('\n', '\n\t\t'), stacklevel=3)
    print(message)
    del message


def safe_division(
        numerator: Union[int, float],
        denominator: Union[int,  float],
) -> float:
    """
    Performs a safe division.

    If denominator is 0 return 0 instead of raising ZeroDivisionError.

    Parameters
    ----------
    numerator : Integer or Float
        DESCRIPTION.
    den : Integer or Float
        DESCRIPTION.

    Returns
    -------
    float
        DESCRIPTION.

    """
    return float(numerator / denominator) if denominator else 0.0


def save_csv(
        dataframe: environment.pandas.DataFrame,
        file_path: str,
        index: bool = False,
):
    """
    Save Pandas DataFrame into a CSV file.

    Parameters
    ----------
    dataframe : Pandas DataFrame
        Pandas DataFrame containing the data to save in the CSV file.
    file_path : str
        Absolute or relative path of the CSV file to save into.

    Returns
    -------
    None.

    """
    dataframe.to_csv(file_path, index=index)


def save_pickle(python_object: object, file_path: str):
    """
    Write a binary dump made by pickle of a Python Object.

    Parameters
    ----------
    python_object : Object
        Python Object to write int a binary dump made by pickle.
    file_path : String
        String with the relative or absolute path of the binary file.

    Returns
    -------
    None.

    """
    with open(file_path, mode='wb') as file:
        environment.pickle.dump(python_object, file)

    del file
    collect_memory()


def transform_redable_byte_scale(number_bytes: int) -> str:
    """
    Tranform a number of bytes into the apropiate unit of the scale to read it.

    Parameters
    ----------
    number_bytes : Integer
        Number of bytes to be transformed.

    Returns
    -------
    String
        Numebr of bytes in the most human redable unit of the scale.

    """
    scale_bytes = ('B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB')
    i = 0
    while number_bytes >= 2 ** 10:
        number_bytes = number_bytes / (2 ** 10)
        i += 1

    return f'{number_bytes} {scale_bytes[i]}'
