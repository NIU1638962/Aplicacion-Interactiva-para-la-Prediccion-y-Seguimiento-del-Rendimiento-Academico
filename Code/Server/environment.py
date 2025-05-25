# -*- coding: utf-8 -*- noqa
"""
Created on Sat May 24 19:44:01 2025

@author: Joel Tapia Salvador
"""
import http
import json
import logging
import os
import sys
import traceback
import urllib
import http.server
import urllib.parse

from copy import deepcopy
from datetime import datetime, timezone

# Server infos
SERVER_IP = ''
SERVER_PORT = 8080

# Time
TIME_EXECUTION = datetime.now(timezone.utc).strftime(
    '%Y-%m-%d--%H-%M-%S-%f--%Z'
)

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

PUBLIC_HTML_FILES_PATH = os.path.join(PROJECT_PATH, 'public')


if USER == '':
    CODE_PATH = ''
    PROJECT_PATH = ''
    LOGS_PATH = ''
    PUBLIC_HTML_FILES_PATH = ''

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
    + f'\nLog level: {LOG_LEVEL_NAME}'
    + f'\nPython version: {PYTHON_VERSION}'
    + f'\nPlatform: {PLATFORM}'
    + f'\nUser: {USER}'
    + f'\nServer IP address: {SERVER_IP}'
    + f'\nServer port: {SERVER_PORT}'
    + f'\nPath to project: "{PROJECT_PATH}"'
    + f'\nPath to code: "{CODE_PATH}"'
    + f'\nPath to logs: "{LOGS_PATH}"'
    + f'\nPath to public HTML files: "{PUBLIC_HTML_FILES_PATH}"'
)


def init():
    # Create paths in case they do not exist
    os.makedirs(PROJECT_PATH, exist_ok=True)
    os.makedirs(LOGS_PATH, exist_ok=True)
    os.makedirs(CODE_PATH, exist_ok=True)
    os.makedirs(PUBLIC_HTML_FILES_PATH, exist_ok=True)

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
    if PYTHON_VERSION[:4] != '3.12':
        ERROR = f'Python 3.12 not found ({PYTHON_VERSION}).'
        logging.critical(ERROR.replace('\n', '\n\t\t'))
        raise SystemError(ERROR)


def finish():
    logging.shutdown()
