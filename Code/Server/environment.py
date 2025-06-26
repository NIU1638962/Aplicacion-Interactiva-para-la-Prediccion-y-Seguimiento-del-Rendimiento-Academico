# -*- coding: utf-8 -*- noqa
"""
Created on Sat May 24 19:44:01 2025

@author: Joel Tapia Salvador
"""
import csv
import gc
import json
import locale
import logging
import os
import pickle
import sys
import traceback

from copy import deepcopy
from datetime import datetime, timezone

# Modules used
import flask
import numpy
import pandas
import psutil
import sklearn
import sklearn.ensemble
import torch
import torchaudio
import torchvision

# Time
TIME_EXECUTION = datetime.now(timezone.utc).strftime(
    '%Y-%m-%d--%H-%M-%S-%f--%Z'
)

# Seed
SEED = 0

# Log level
LOG_LEVEL = logging.DEBUG
LOG_LEVEL_NAME = logging.getLevelName(LOG_LEVEL)

# Python version
PYTHON_VERSION = '{0}.{1}.{2}'.format(
    *sys.version_info.__getnewargs__()[0][:3]
)

# Platform
PLATFORM = sys.platform.lower()

# User
USER = ''

if PLATFORM == 'win32':  # Windows
    USER = os.getenv('USERNAME')
else:  # Unix-like platforms
    USER = os.getenv('USER')

# Paths
CODE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = deepcopy(CODE_PATH)
LOGS_PATH = os.path.join(PROJECT_PATH, 'logs')

CONFIGURATION_FILE_PATH = os.path.join(PROJECT_PATH, 'configuration.json')


if USER == '':
    CODE_PATH = ''
    PROJECT_PATH = ''
    LOGS_PATH = ''

    CONFIGURATION_FILE_PATH = ''


# CPU cores
NUMBER_PHYSICAL_PROCESSORS = psutil.cpu_count(logical=False)
NUMBER_LOGICAL_PROCESSORS = psutil.cpu_count(logical=True)


# Torch CUDA and device
CUDA_AVAILABLE = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda:0' if CUDA_AVAILABLE else 'cpu')

# Prints
try:
    TERMINAL_WIDTH = os.get_terminal_size().columns
except OSError:
    TERMINAL_WIDTH = 80

SECTION = '='
SECTION_LINE = SECTION * TERMINAL_WIDTH
SEPARATOR = '-'
SEPARATOR_LINE = SEPARATOR * TERMINAL_WIDTH
MARKER = '*'

EXECUTION_INFORMATION = (
    SECTION_LINE
    + '\nENVIRONMENT INFO\n'
    + SEPARATOR_LINE
    + f'\nTime execution: {TIME_EXECUTION}'
    + f'\nSeed: {SEED}'
    + f'\nLog level: {LOG_LEVEL_NAME}'
    + f'\nPython version: {PYTHON_VERSION}'
    + f'\nPlatform: {PLATFORM}'
    + f'\nUser: {USER}'
    + f'\nPath to project: "{PROJECT_PATH}"'
    + f'\nPath to code: "{CODE_PATH}"'
    + f'\nPath to logs: "{LOGS_PATH}"'
    + f'\nPath to configuration path: "{CONFIGURATION_FILE_PATH}'
    + f'\nPhysical Processors: {NUMBER_PHYSICAL_PROCESSORS}'
    + f'\nLogical Processors: {NUMBER_LOGICAL_PROCESSORS}'
    + f'\nCuda available: {CUDA_AVAILABLE}'
    + f'\nTorch device: {str(TORCH_DEVICE).replace(":", " ")}'
    + f' ({torch.cuda.get_device_properties(TORCH_DEVICE).name})' if CUDA_AVAILABLE else ''
)


def init():
    # Create paths in case they do not exist
    os.makedirs(PROJECT_PATH, exist_ok=True)
    os.makedirs(LOGS_PATH, exist_ok=True)
    os.makedirs(CODE_PATH, exist_ok=True)

    # Logging set up
    logging.basicConfig(
        filename=os.path.join(
            LOGS_PATH,
            f'{TIME_EXECUTION}--{PLATFORM}--{USER}--{LOG_LEVEL_NAME}.log',
        ),
        filemode='w',
        level=LOG_LEVEL,
        force=True,
        format=(
            '[%(asctime)s] %(levelname)s:\n\tModule: "%(module)s"\n\t'
            + 'Function: "%(funcName)s"\n\tLine: %(lineno)d\n\tLog:\n\t\t'
            + '%(message)s\n'
        ),
    )

    logging.info(EXECUTION_INFORMATION.replace('\n', '\n\t\t'))
    print(EXECUTION_INFORMATION)

    # Test versions
    if PYTHON_VERSION != '3.8.10':
        ERROR = f'Python 3.8.10 not found ({PYTHON_VERSION}).'
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise SystemError(ERROR)

    if flask.__version__ != '3.0.3':
        ERROR = f'Module flask 3.0.3 not found ({flask.__version__}).'
        logging.critical(ERROR.replace('\n', '\n\t\t'))

    if numpy.__version__ != '1.24.4':
        ERROR = f'Module numpy 1.24.4 not found ({numpy.__version__}).'
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise ModuleNotFoundError(ERROR)

    if pandas.__version__ != '2.0.3':
        ERROR = f'Module pandas 2.0.3 not found ({pandas.__version__}).'
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise ModuleNotFoundError(ERROR)

    if psutil.__version__ != '6.0.0':
        ERROR = f'Module psutil 6.0.0 not found ({psutil.__version__}).'
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise ModuleNotFoundError(ERROR)

    if sklearn.__version__ != '1.3.2':
        ERROR = f'Module sklearn 1.3.2 not found ({sklearn.__version__}).'
        logging.critical(ERROR.replace('\n', '\n\t\t'))

    if torch.__version__ not in ('2.4.1', '2.4.1+cu121'):
        ERROR = f'Module torch 2.4.1 not found ({torch.__version__}).'
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise ModuleNotFoundError(ERROR)

    if torchaudio.__version__ not in ('2.4.1', '2.4.1+cu121'):
        ERROR = (
            f'Module torchaudio 2.4.1 not found ({torchaudio.__version__}).'
        )
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise ModuleNotFoundError(ERROR)

    if torchvision.__version__ not in ('0.19.1', '0.19.1+cu121'):
        ERROR = (
            f'Module torchvision 0.19.1 not found ({torchvision.__version__}).'
        )
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise ModuleNotFoundError(ERROR)


def finish():
    logging.shutdown()
