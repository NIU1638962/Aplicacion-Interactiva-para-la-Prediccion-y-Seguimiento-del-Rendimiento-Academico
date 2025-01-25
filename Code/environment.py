# -*- coding: utf-8 -*- noqa
"""
Created on Thu Jan 23 22:28:45 2025

@author: Joel Tapia Salvador
"""
import logging
import os
import sys

from datetime import datetime, timezone

TIME_EXECUTION = datetime.now(timezone.utc).strftime(
    '%Y-%m-%d--%H-%M-%S-%f--%Z'
)

LOG_LEVEL = logging.DEBUG
log_level_name = logging.getLevelName(LOG_LEVEL)

PLATFORM = sys.platform.lower()

USER = ''

CODE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = os.path.dirname(CODE_PATH)

if PLATFORM == 'win32':  # Windows
    USER = os.getenv('USERNAME')
else:  # Unix-like platforms
    USER = os.getenv('USER')

USER.lower()

if USER == '':
    DATA_PATH = ''
    LOGS_PATH = ''
    PICKLE_PATH = ''
    REQUIREMENTS_PATH = ''
    RESULTS_PATH = ''
    TRAINED_MODELS_PATH = ''
else:
    DATA_PATH = os.path.join(PROJECT_PATH, 'Data')
    LOGS_PATH = os.path.join(PROJECT_PATH, 'Logs')
    PICKLE_PATH = os.path.join(PROJECT_PATH, 'Pickle')
    REQUIREMENTS_PATH = os.path.join(PROJECT_PATH, 'Requirements')
    RESULTS_PATH = os.path.join(PROJECT_PATH, 'Results')
    TRAINED_MODELS_PATH = os.path.join(PROJECT_PATH, 'Trained Models')

# Create paths in case do not exist
os.makedirs(PROJECT_PATH, exist_ok=True)
os.makedirs(CODE_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)
os.makedirs(PICKLE_PATH, exist_ok=True)
os.makedirs(REQUIREMENTS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

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

)

print(execution_information)

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
