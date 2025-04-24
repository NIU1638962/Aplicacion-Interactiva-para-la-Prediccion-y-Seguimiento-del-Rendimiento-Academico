# -*- coding: utf-8 -*- noqa
"""
Created on Sun Apr 20 21:08:59 2025

@author: Joel Tapia Salvador
"""
import environment
import utils

class Dataset(environment.torch.utils.data.Dataset):
    __slots__ = ('__inputs', '__targets')
    
    def __init__(self, inputs, targets):
        self.__inputs = inputs
        self.__targets = targets
        
    def __getitem__(self, index):
        return self.__inputs[index], self.__targets[index]
    
    def __len__(self):
        return len(self.__inputs)
    
    @property
    def inputs(self):
        return self.__inputs
    
    @property 
    def targets(self):
        return self.__targets
        
        

def get_datasets(mode):
    utils.print_message(
        environment.SECTION_LINE
        + '\nGETTING DATASETS'    
    )
    
    utils.print_message(
        environment.SEPARATOR_LINE
        + '\nLoading Dataset'
    )
    dataframe = utils.load_csv(
        environment.os.path.join(
            environment.DATA_PATH,
            'dataset.csv',
        )    
    )
    utils.print_message('Dataset Loaded')
    
    utils.print_message(
        environment.SEPARATOR_LINE
        + '\nGenerating Datasets'
    )
    columns = dataframe.columns.drop(['target'])
    
    dataset = Dataset(
        environment.torch.from_numpy(
            dataframe[columns].to_numpy(),
        ),
        environment.torch.from_numpy(
            dataframe['target'].astype('int64').to_numpy(),
        ),
    )
        
    
    if mode == "experiment_1":
        k_fold = environment.sklearn.model_selection.StratifiedKFold(
            n_splits=20,
            shuffle=True,
            random_state=environment.SEED,
        )
        _, mask = next(k_fold.split(dataset.inputs, dataset.targets))
        
        del _
        utils.collect_memory()
        
        dataset = Dataset(*dataset[mask])
        utils.print_message('Datasets Generated')
        
        return dataset
    
    
def k_fold(dataset, number_folds):
    k_fold = environment.sklearn.model_selection.StratifiedKFold(
        n_splits=number_folds,
        shuffle=True,
        random_state=environment.SEED,
    )
    
    for fold, (train_mask, validation_mask) in  enumerate(
            k_fold.split(dataset.inputs, dataset.targets)
        ):
        fold += 1
        train_dataset = Dataset(*dataset[train_mask])
        validation_dataset = Dataset(*dataset[validation_mask])
        
        del train_mask, validation_mask
        utils.collect_memory()
        
        yield fold, train_dataset, validation_dataset
        
        
        
        
    
    
    
    
    