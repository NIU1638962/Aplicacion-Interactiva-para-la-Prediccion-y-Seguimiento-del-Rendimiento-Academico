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

    Print the memory information (total memory, free memory, used memory) in
    MiB of the CUDA device being used and return such information in bytes.

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

    free_memory_mib = free_memory / 2 ** 20
    total_memory_mib = total_memory / 2 ** 20
    used_memory_mib = used_memory / 2 ** 20

    verbose_memory_info_mib = (
        f'Total CUDA Memory: {total_memory_mib}MiB'
        + f'\nUsed CUDA Memory: {used_memory_mib}MiB'
        + f'\nFree CUDA Memory: {free_memory_mib}MiB'
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

    Print the memory information (total memory, free memory, used memory) in
    MiB of the system and return such information in bytes.

    Returns
    -------
    total_memory : Integer
        Number of bytes of the total memory of the system.
    used_memory : Integer
        Number of bytes currently used of the system.
    free_memory : Integer
        Number of bytes currently free of the system.

    """
    total_memory, free_memory, _, _, _ = environment.psutil.virtual_memory()

    del _
    collect_memory()

    used_memory = total_memory - free_memory

    free_memory_mib = free_memory / 2 ** 20
    total_memory_mib = total_memory / 2 ** 20
    used_memory_mib = used_memory / 2 ** 20

    verbose_memory_info_mib = (
        f'Total System Memory: {total_memory_mib}MiB'
        + f'\nUsed System Memory: {used_memory_mib}MiB'
        + f'\nFree System Memory: {free_memory_mib}MiB'
    )

    environment.logging.info(verbose_memory_info_mib.replace('\n', '\n\t\t'))

    print(verbose_memory_info_mib)

    return total_memory, used_memory, free_memory
