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

    verbose_memory_info_mib = (
        f'Total CUDA Memory: {redable_total_memory}'
        + f'\nUsed CUDA Memory: {redable_used_memory}'
        + f'\nFree CUDA Memory: {redable_free_memory}'
    )

    environment.logging.info(verbose_memory_info_mib.replace('\n', '\n\t\t'))

    print(verbose_memory_info_mib)

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

    del memory
    collect_memory()

    used_memory = total_memory - free_memory

    redable_free_memory = transform_redable_byte_scale(free_memory)
    redable_total_memory = transform_redable_byte_scale(total_memory)
    redable_used_memory = transform_redable_byte_scale(used_memory)

    verbose_memory_info_mib = (
        f'Total System Memory: {redable_total_memory}'
        + f'\nUsed System Memory: {redable_used_memory}'
        + f'\nFree System Memory: {redable_free_memory}'
    )

    environment.logging.info(verbose_memory_info_mib.replace('\n', '\n\t\t'))

    print(verbose_memory_info_mib)

    return total_memory, used_memory, free_memory


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
