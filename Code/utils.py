# -*- coding: utf-8 -*- noqa
"""
Created on Tue Mar 25 21:29:37 2025

@author: Joel Tapia Salvador
"""
import environment

from typing import Tuple


def cuda_memory() -> Tuple[int, int, int]:
    """
    Get information about CUDA's device memory.

    Print the memory information (total memory, free memory, used memory) in MB
    of the CUDA device being used and return such information in bytes.

    Returns
    -------
    total_memory : integer
        Number of bytes of the total memory of the CUDA device.
    used_memory : integer
        Number of bytes of CUDA device currently used.
    free_memory : TYPE
        Number of bytes of CUDA device currently free.

    """
    total_memory = 0
    free_memory = 0

    if environment.CUDA_AVAILABLE and environment.TORCH_DEVICE.type == 'cuda':
        free_memory, total_memory = environment.torch.cuda.mem_get_info(
            environment.TORCH_DEVICE
        )

    used_memory = total_memory - free_memory

    free_memory_mb = free_memory / 1024 ** 2
    total_memory_mb = total_memory / 1024 ** 2
    used_memory_mb = used_memory / 1024 ** 2

    memory_info_mb = (
        f'Total Memory: {total_memory_mb}MB'
        + f'\nUsed Memory: {used_memory_mb}MB'
        + f'\nFree Memory: {free_memory_mb}MB'
    )

    environment.logging.info(memory_info_mb.replace('\n', '\n\t\t'))

    print(memory_info_mb)

    return total_memory, used_memory, free_memory
