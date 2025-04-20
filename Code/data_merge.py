# -*- coding: utf-8 -*- noqa
"""
Created on Sat Apr 19 23:07:16 2025

@author: Joel Tapia Salvador
"""
import environment
import utils

def merge_data(number_grades):
    environment.NUMBER_GRADES = number_grades

    subfolders = tuple([
        file.path for file in environment.os.scandir(environment.DATA_PATH) if file.is_dir()
    ])
    
    with open(
            environment.os.path.join(
                environment.DATA_PATH,
                'general_mapping.json',
            )
    ) as file:
        general_mapping = dict(sorted(environment.json.load(file).items()))
    
    del file
    utils.collect_memory()
    
    dataframe = environment.pandas.DataFrame(columns=general_mapping.keys())
    
    for subfolder in subfolders[:3]:
        print(environment.SEPARATOR_LINE)
        print(f'Data: "{environment.os.path.split(subfolder)[1]}"')
    
        file_path = environment.os.path.join(subfolder, 'data.csv')
        
        data = utils.load_csv(file_path)
    
        mapping = utils.module_from_file(
            "mapping",
            environment.os.path.join(subfolder, 'mapping_data.py'),
        )
    
        for column_name, info in mapping.mapping.items():
            if info['drop']:
                data = data.drop(columns=[column_name])
            else:
                data[column_name] = data[column_name].map(info['map_values'])
                data = data.rename(columns={column_name: info['new_name']})
    
        del column_name, info, mapping
        utils.collect_memory()
    
        for column_name, info in general_mapping.items():
            if column_name not in data.columns:
                data[column_name] = info['not known']
    
        del column_name, info
        utils.collect_memory()
    
        data = data[sorted(data.columns)]
    
        print(data.info())
        print(data.shape)
    
        assert data.shape[1] == dataframe.shape[1]
    
        dataframe = environment.pandas.concat([dataframe, data])
    
        del data
        utils.collect_memory()
        
        print('Dataframe:')
        print(dataframe.info())
        print(dataframe.shape)
        
    
    dataframe['target'] = environment.pandas.cut(
        dataframe['target'], 
        bins=environment.NUMBER_GRADES,
        labels=[i for i in range(environment.NUMBER_GRADES)],
    )
    
    print('Final Dataframe:')
    print(dataframe.info())
    print(dataframe.shape)
    
    print(dataframe)
        
    utils.save_csv(
        dataframe,
        environment.os.path.join(environment.DATA_PATH, 'dataset.csv'),
    )