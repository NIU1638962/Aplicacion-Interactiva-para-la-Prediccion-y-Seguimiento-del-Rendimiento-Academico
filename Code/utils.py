# -*- coding: utf-8 -*- noqa
"""
Created on Tue Mar 25 21:29:37 2025

@author: Joel Tapia Salvador
"""
from typing import Tuple

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


def load_csv(file_path):
    with open(file_path) as csv_file:
        dialect = environment.csv.Sniffer().sniff(csv_file.readline())

    del csv_file
    collect_memory()

    data = environment.pandas.read_csv(file_path, sep=dialect.delimiter)

    del dialect
    collect_memory()
    
    return data


def load_json(file_path):
    with open(file_path) as file:
        json_object = environment.json.load(file)
        
    del file
    collect_memory()
    
    return json_object


def module_from_file(module_name, file_path):
    spec = environment.importlib.util.spec_from_file_location(
        module_name, file_path
    )
    module = environment.importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def print_message(message):
    environment.logging.info(message)
    print(message)
    del message

def save_csv(dataframe, file_path):
    dataframe.to_csv(file_path, index=False)


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
