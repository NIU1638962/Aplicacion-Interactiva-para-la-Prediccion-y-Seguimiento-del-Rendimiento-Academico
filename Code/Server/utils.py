# -*- coding: utf-8 -*- noqa
"""
Created on Sat May 24 21:53:36 2025

@author: Joel Tapia Salvador
"""
import environment

from typing import Union


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


def load_csv(
    file_path: str,
    encoding: str = environment.locale.getpreferredencoding(),
    header='infer',
    skip_blank_lines: bool = True,
    low_memory: bool = True,
    index_column=None,
) -> environment.pandas.DataFrame:
    """
    Load and read a CSV file into a Pandas DataFrame.

    Parameters
    ----------
    file_path : String
        String with the relative or absolute path of the CSV file.
    encoding : String, optional
        String with the encoding used on the CSV file. The default is
        locale.getpreferredencoding().

    Returns
    -------
    data : Pandas DataFrame
        Pandas Dataframe containing the CSV data.

    """
    with open(file_path, mode='r', encoding=encoding) as csv_file:
        dialect = environment.csv.Sniffer().sniff(csv_file.readline())

    del csv_file
    collect_memory()

    data = environment.pandas.read_csv(
        file_path,
        sep=dialect.delimiter,
        header=header,
        skip_blank_lines=skip_blank_lines,
        low_memory=low_memory,
        index_col=index_column,
    )

    del dialect
    collect_memory()

    return data


def load_json(
    file_path: str,
    encoding: str = environment.locale.getpreferredencoding(),
) -> object:
    """
    Load a JSON file into a Python Object.

    Parameters
    ----------
    file_path : String
        String with the relative or absolute path of the JSON file.
    encoding : String, optional
        String with the encoding used on the JSON file. The default is
        locale.getpreferredencoding().

    Returns
    -------
    python_object : Object
        Python Object containing the JSON data.

    """
    with open(file_path, mode='r', encoding=encoding) as file:
        python_object = environment.json.load(file)

    del file

    return python_object


def load_pickle(file_path: str) -> object:
    """
    Read a binary dump made by pickle of a Python Object.

    Parameters
    ----------
    file_path : Sting
        String with the relative or absolute path of the binary file.

    Returns
    -------
    python_object : Object
        Python Object read from the binary dump made by pickle.

    """
    with open(file_path, mode='rb') as file:
        python_object = environment.pickle.load(file)

    del file

    return python_object


def print_error(error: Union[str, None] = None, print_stack: bool = True):
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


def safe_division(
        numerator: Union[int, float],
        denominator: Union[int,  float],
        zero_division_result: float = 0.0,
) -> float:
    """
    Performs a safe division.

    If denominator is 0 return zero_division_result instead of raising ZeroDivisionError.

    Parameters
    ----------
    numerator : Integer or Float
        DESCRIPTION.
    den : Integer or Float
        DESCRIPTION.

    Returns
    -------
    float
        DESCRIPTION.

    """
    return float(numerator / denominator) if denominator else zero_division_result


def save_pickle(python_object: object, file_path: str):
    """
    Write a binary dump made by pickle of a Python Object.

    Parameters
    ----------
    python_object : Object
        Python Object to write int a binary dump made by pickle.
    file_path : String
        String with the relative or absolute path of the binary file.

    Returns
    -------
    None.

    """
    with open(file_path, mode='wb') as file:
        environment.pickle.dump(python_object, file)

    del file
    collect_memory()


def un_string_parameters(data):
    new_data = {}
    for key, value in data.items():
        try:
            new_data[key] = int(value)
        except ValueError:
            try:
                new_data[key] = float(value)
            except ValueError:
                if value.lower() in ('none', 'null'):
                    new_data[key] = None
                else:
                    new_data[key] = value

    return new_data
