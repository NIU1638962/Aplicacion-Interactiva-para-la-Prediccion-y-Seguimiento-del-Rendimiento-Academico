# -*- coding: utf-8 -*- noqa
"""
Created on Sat Apr 19 23:07:16 2025

@author: Joel Tapia Salvador
"""
import environment
import utils

def merge_data():
    utils.print_message(
        environment.SECTION_LINE 
        + '\nMERGING DATA'
    )

    subfolders = tuple([
        file.path for file in environment.os.scandir(environment.DATA_PATH) if file.is_dir()
    ])
    
    general_mapping = dict(
        sorted(
            utils.load_json(
                environment.os.path.join(
                    environment.DATA_PATH,
                    'general_mapping.json',
                )
            ).items()
        )
    )
    
    dataframe = environment.pandas.DataFrame(columns=general_mapping.keys())
    
    for subfolder in subfolders[:3]:
        utils.print_message(
            environment.SEPARATOR_LINE
            +'\nReading data from: '
            + f'"{environment.os.path.split(subfolder)[1]}"'
        )
        
        data = utils.load_csv(environment.os.path.join(subfolder, 'data.csv'))
    
        data_mapping = utils.module_from_file(
            "mapping",
            environment.os.path.join(subfolder, 'mapping_data.py'),
        )
    
        for column_name, info in data_mapping.mapping.items():
            if info['drop']:
                data = data.drop(columns=[column_name])
            else:
                data[column_name] = data[column_name].map(info['map_values'])
                data = data.rename(columns={column_name: info['new_name']})
    
        del column_name, info, data_mapping
        utils.collect_memory()
    
        for column_name, info in general_mapping.items():
            if column_name not in data.columns:
                data[column_name] = info['not known']
    
        del column_name, info
        utils.collect_memory()
    
        data = data[sorted(data.columns)]
    
        utils.print_message(
            f'{environment.SECTION * 3}Data Read{environment.SECTION * 3}\n'
            + f'{environment.SEPARATOR * 3}Resume{environment.SEPARATOR * 3}\n'
            + f'{data.describe(include="all")}'
            + f'\n{environment.SEPARATOR * 3}Non nulls{environment.SEPARATOR * 3}\n'
            + f'{environment.pandas.io.formats.info.DataFrameInfo(data).non_null_counts}'
            + f'\nShape: [{data.shape[0]} rows x {data.shape[1]} columns]\n'
            + 'Memory used: '
            + f'{utils.transform_redable_byte_scale(data.memory_usage(deep=True).sum())}'
        )
        
        assert data.shape[1] == dataframe.shape[1], 'Data read and final dataframe do not match in shape.'
    
        dataframe = environment.pandas.concat([dataframe, data])
    
        del data
        utils.collect_memory()
        
        utils.print_message(
            f'{environment.SECTION * 3}Dataframe{environment.SECTION * 3}\n'
            + f'{environment.SEPARATOR * 3}Resume{environment.SEPARATOR * 3}\n'
            + f'{dataframe.describe(include="all")}'
            + f'\n{environment.SEPARATOR * 3}Non nulls{environment.SEPARATOR * 3}\n'
            + f'{environment.pandas.io.formats.info.DataFrameInfo(dataframe).non_null_counts}'
            + f'\nShape: [{dataframe.shape[0]} rows x {dataframe.shape[1]} columns]\n'
            + 'Memory used: '
            + f'{utils.transform_redable_byte_scale(dataframe.memory_usage(deep=True).sum())}'
        )
        
    
    dataframe['target'] = environment.pandas.cut(
        dataframe['target'], 
        bins=environment.NUMBER_GRADES,
        labels=[i for i in range(environment.NUMBER_GRADES)],
    )
    
    utils.print_message(
        environment.SEPARATOR_LINE
        + '\nFinal Dataframe\n'
        + f'{environment.SECTION * 3}Dataframe{environment.SECTION * 3}\n'
        + f'{environment.SEPARATOR * 3}Resume{environment.SEPARATOR * 3}\n'
        + f'{dataframe.describe(include="all")}'
        + f'\n{environment.SEPARATOR * 3}Non nulls{environment.SEPARATOR * 3}\n'
        + f'{environment.pandas.io.formats.info.DataFrameInfo(dataframe).non_null_counts}'
        + f'\nShape: [{dataframe.shape[0]} rows x {dataframe.shape[1]} columns]\n'
        + 'Memory used: '
        + f'{utils.transform_redable_byte_scale(dataframe.memory_usage(deep=True).sum())}'
    )
        
    utils.save_csv(
        dataframe,
        environment.os.path.join(environment.DATA_PATH, 'dataset.csv'),
    )