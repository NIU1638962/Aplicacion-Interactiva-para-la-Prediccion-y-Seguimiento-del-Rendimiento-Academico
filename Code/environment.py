# -*- coding: utf-8 -*- noqa
"""
Created on Thu Jan 23 22:28:45 2025

@author: Joel Tapia Salvador
"""
import csv
import warnings
import gc
import hashlib
import importlib.util
import json
import locale
import logging
import math
import os
import pickle
import sys
import types

from copy import deepcopy
from datetime import datetime, timezone
from time import time

# Modules used
import matplotlib
import numpy
import pandas
import psutil
import sklearn
import sklearn.ensemble
import sklearn.model_selection
import sklearn.linear_model
import sklearn.tree
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
PROJECT_PATH = os.path.dirname(CODE_PATH)

if USER == '':
    DATA_PATH = ''
    LOGS_PATH = ''
    PICKLE_PATH = ''
    REQUIREMENTS_PATH = ''
    RESULTS_PATH = ''
    TRAINED_MODELS_PATH = ''
elif USER.lower() == 'jtapia':
    storage_path, repository_name = os.path.split(
        PROJECT_PATH
    )
    storage_path = os.path.join(
        os.path.dirname(
            os.path.dirname(
                storage_path
            )
        ),
        USER,
    )
    DATA_PATH = os.path.join(
        storage_path,
        repository_name,
        'Data',
    )
    LOGS_PATH = os.path.join(storage_path, 'Logs')
    PICKLE_PATH = os.path.join(storage_path, 'Pickle')
    REQUIREMENTS_PATH = os.path.join(PROJECT_PATH, 'Requirements')
    RESULTS_PATH = os.path.join(storage_path, 'Results')
    TRAINED_MODELS_PATH = os.path.join(storage_path, 'Trained Models')
    del storage_path, repository_name
    gc.collect()
else:
    DATA_PATH = os.path.join(PROJECT_PATH, 'Data')
    LOGS_PATH = os.path.join(PROJECT_PATH, 'Logs')
    PICKLE_PATH = os.path.join(PROJECT_PATH, 'Pickle')
    REQUIREMENTS_PATH = os.path.join(PROJECT_PATH, 'Requirements')
    RESULTS_PATH = os.path.join(PROJECT_PATH, 'Results')
    TRAINED_MODELS_PATH = os.path.join(PROJECT_PATH, 'Trained Models')

# Create paths in case they do not exist
os.makedirs(PROJECT_PATH, exist_ok=True)
os.makedirs(CODE_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)
os.makedirs(PICKLE_PATH, exist_ok=True)
os.makedirs(REQUIREMENTS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(TRAINED_MODELS_PATH, exist_ok=True)

# CPU cores
NUMBER_PHYSICAL_PROCESSORS = psutil.cpu_count(logical=False)
NUMBER_LOGICAL_PROCESSORS = psutil.cpu_count(logical=True)


# Torch CUDA and device
CUDA_AVAILABLE = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda:0' if CUDA_AVAILABLE else 'cpu')

# Prints
TERMINAL_WIDTH = os.get_terminal_size().columns
SECTION = '='
SECTION_LINE = SECTION * TERMINAL_WIDTH
SEPARATOR = '-'
SEPARATOR_LINE = SEPARATOR * TERMINAL_WIDTH

EXECUTION_INFORMATION = (
    SECTION_LINE
    + '\nENVIRONMENT INFO\n'
    + SEPARATOR_LINE
    + f'\nTime execution: {TIME_EXECUTION}'
    + f'\nSeed: {SEED}'
    + f'\nLog level: {LOG_LEVEL_NAME}'
    + f'\nPlatform: {PLATFORM}'
    + f'\nUser: {USER}'
    + f'\nPath to project: "{PROJECT_PATH}"'
    + f'\nPath to code: "{CODE_PATH}"'
    + f'\nPath to data: "{DATA_PATH}"'
    + f'\nPath to logs: "{LOGS_PATH}"'
    + f'\nPath to pickle: "{PICKLE_PATH}"'
    + f'\nPath to requirements: "{REQUIREMENTS_PATH}"'
    + f'\nPath to results: "{RESULTS_PATH}"'
    + f'\nPath to trained models: "{TRAINED_MODELS_PATH}"'
    + f'\nPhysical Processors: {NUMBER_PHYSICAL_PROCESSORS}'
    + f'\nLogical Processors: {NUMBER_LOGICAL_PROCESSORS}'
    + f'\nCuda available: {CUDA_AVAILABLE}'
    + f'\nTorch device: {str(TORCH_DEVICE).replace(":", " ")}'
    + f' ({torch.cuda.get_device_properties(TORCH_DEVICE).name})' if CUDA_AVAILABLE else ''
)

# Logging set up
logging.basicConfig(
    filename=os.path.join(
        LOGS_PATH,
        f'{TIME_EXECUTION}--{PLATFORM}--{USER}--{LOG_LEVEL_NAME}.log',
    ),
    filemode='w',
    level=LOG_LEVEL,
    force=True,
    format='[%(asctime)s] %(levelname)s:\n\tModule: "%(module)s"\n\t' +
    'Function: "%(funcName)s"\n\tLine: %(lineno)d\n\tLog:\n\t\t%(message)s\n',
)

logging.info(EXECUTION_INFORMATION.replace('\n', '\n\t\t'))
print(EXECUTION_INFORMATION)

# Test modules
if matplotlib.__version__ != '3.7.3':
    ERROR = f'Module matplotlib 3.7.3 not found ({matplotlib.__version__}).'
    logging.critical(ERROR.replace('\n', '\n\t\t'))
    raise ModuleNotFoundError(ERROR)
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
    raise ModuleNotFoundError(ERROR)
if torch.__version__ not in ('2.4.1', '2.4.1+cu121'):
    ERROR = f'Module torch 2.4.1 not found ({torch.__version__}).'
    logging.critical(ERROR.replace('\n', '\n\t\t'))
    raise ModuleNotFoundError(ERROR)
if torchaudio.__version__ not in ('2.4.1', '2.4.1+cu121'):
    ERROR = f'Module torchaudio 2.4.1 not found ({torchaudio.__version__}).'
    logging.critical(ERROR.replace('\n', '\n\t\t'))
    raise ModuleNotFoundError(ERROR)
if torchvision.__version__ not in ('0.19.1', '0.19.1+cu121'):
    ERROR = f'Module torchvision 0.19.1 not found ({torchvision.__version__}).'
    logging.critical(ERROR.replace('\n', '\n\t\t'))
    raise ModuleNotFoundError(ERROR)

# Setting modules
matplotlib.pyplot.ioff()
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)
# warnings.filterwarnings("ignore", category=UserWarning)
