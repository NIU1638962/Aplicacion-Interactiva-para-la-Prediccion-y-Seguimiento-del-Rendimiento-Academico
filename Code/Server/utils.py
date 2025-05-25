# -*- coding: utf-8 -*- noqa
"""
Created on Sat May 24 21:53:36 2025

@author: Joel Tapia Salvador
"""
import environment


def formated_traceback_stack() -> str:
    """
    Get Formated Traceback Stack.

    Returns the Formated Traceback Stack untill the point before calling this
    funtion and formats it to the style of the logs.

    Returns
    -------
    string
        Formated Traceback Stack.

    """
    return 'Traceback (most recent call last):' + (
        '\n' + ''.join(environment.traceback.format_stack()[:-2])[:-1]
    )


def print_error(error: str | None = None, print_stack: bool = True):
    if print_stack:
        stack = formated_traceback_stack()

        if error is None or error == '':
            message = stack
        else:
            message = error + '\n' + stack
        del error, stack

    else:
        message = error
        del error

    environment.logging.error(message.replace('\n  ', '\n\t\t'), stacklevel=3)
    print(message)
    del message


def print_message(message: str = ''):
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
