# -*- coding: utf-8 -*- noqa
"""
Created on Thu Jan 23 22:28:45 2025

@author: Joel Tapia Salvador
"""
import logging
import os
import sys

from datetime import datetime, timezone

# Modules used
import matplotlib
import numpy
import pandas
import sklearn
import torch
import torchaudio
import torchvision

# Time
TIME_EXECUTION = datetime.now(timezone.utc).strftime(
    '%Y-%m-%d--%H-%M-%S-%f--%Z'
)

# Log level
LOG_LEVEL = logging.DEBUG
log_level_name = logging.getLevelName(LOG_LEVEL)

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
    del storage_path. repository_name
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
os.makedirs(RESULTS_PATH, exist_ok=True)

# Torch CUDA and device
CUDA_AVAILABLE = torch.cuda.is_available()
TORCH_DEVICE = torch.device('cuda:0' if CUDA_AVAILABLE else 'cpu')

execution_information = (
    f'Time execution: {TIME_EXECUTION}'
    + f'\nLog level: {log_level_name}'
    + f'\nPlatform: {PLATFORM}'
    + f'\nUser: {USER}'
    + f'\nPath to project: "{PROJECT_PATH}"'
    + f'\nPath to code: "{CODE_PATH}"'
    + f'\nPath to data: "{DATA_PATH}"'
    + f'\nPath to logs: "{LOGS_PATH}"'
    + f'\nPath to pickle: "{PICKLE_PATH}"'
    + f'\nPath to requirements: "{REQUIREMENTS_PATH}"'
    + f'\nPath to results: "{RESULTS_PATH}"'
    + f'\nPath to trained models: "{RESULTS_PATH}"'
    + f'\nCuda available: {CUDA_AVAILABLE}'
    + f'\nTorch device: {str(TORCH_DEVICE).replace(":", " ")}'
)

# Logging set up
logging.basicConfig(
    filename=os.path.join(
        LOGS_PATH,
        f'{TIME_EXECUTION}--{PLATFORM}--{USER}--{log_level_name}.log',
    ),
    filemode='w',
    level=LOG_LEVEL,
    force=True,
    format='[%(asctime)s] %(levelname)s:\n\tModule: "%(module)s"\n\t' +
    'Function: "%(funcName)s"\n\tLine: %(lineno)d\n\tLog:\n\t\t%(message)s\n',
)

logging.info(execution_information.replace('\n', '\n\t\t'))

print(execution_information)

# Test modules
if matplotlib.__version__ != '3.7.3':
    ERROR = 'Module matplotlib 3.7.3 not found.'
    logging.critical(ERROR.replace('\n', '\n\t\t'))
    raise ModuleNotFoundError(ERROR)
if numpy.__version__ != '1.24.4':
    ERROR = 'Module numpy 1.24.4 not found.'
    logging.critical(ERROR.replace('\n', '\n\t\t'))
    raise ModuleNotFoundError(ERROR)
if pandas.__version__ != '2.0.3':
    ERROR = 'Module pandas 2.0.3 not found.'
    logging.critical(ERROR.replace('\n', '\n\t\t'))
    raise ModuleNotFoundError(ERROR)
if sklearn.__version__ != '1.3.2':
    ERROR = 'Module sklearn 1.3.2 not found.'
    logging.critical(ERROR.replace('\n', '\n\t\t'))
    raise ModuleNotFoundError(ERROR)
if torch.__version__ != '2.4.1':
    ERROR = 'Module torch 2.4.1 not found.'
    logging.critical(ERROR.replace('\n', '\n\t\t'))
    raise ModuleNotFoundError(ERROR)
if torchaudio.__version__ != '2.4.1':
    ERROR = 'Module torchaudio 2.4.1 not found.'
    logging.critical(ERROR.replace('\n', '\n\t\t'))
    raise ModuleNotFoundError(ERROR)
if torchvision.__version__ != '0.19.1':
    ERROR = 'Module torchvision 0.19.1 not found.'
    logging.critical(ERROR.replace('\n', '\n\t\t'))
    raise ModuleNotFoundError(ERROR)
