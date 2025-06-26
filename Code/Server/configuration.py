# -*- coding: utf-8 -*- noqa
"""
Created on Sun Jun 22 23:24:02 2025

@author: JoelT
"""
import environment
import utils

class_order_file_path = ''
fields_file_path = ''
model_file_path = ''
original_dataset_file_path = ''
server_ip = ''
server_port = 0


def reload_config():
    configuration = utils.load_json(environment.CONFIGURATION_FILE_PATH)

    global class_order_file_path
    global fields_file_path
    global model_file_path
    global original_dataset_file_path
    global server_ip
    global server_port

    class_order_file_path = environment.os.path.join(
        environment.PROJECT_PATH,
        *configuration['class_order_file_path'],
    )
    fields_file_path = environment.os.path.join(
        environment.PROJECT_PATH,
        *configuration['fields_file_path'],
    )
    model_file_path = environment.os.path.join(
        environment.PROJECT_PATH,
        *configuration['model_file_path'],
    )
    original_dataset_file_path = environment.os.path.join(
        environment.PROJECT_PATH,
        *configuration['original_dataset_file_path']
    )
    server_ip = configuration['server_ip']
    server_port = configuration['server_port']
