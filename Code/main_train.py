# -*- coding: utf-8 -*- noqa
"""
Created on Sun Apr 20 21:04:02 2025

@author: Joel Tapia Salvador
"""
import environment
import utils
import models_sklearn
import train_sklearn

from data_merge import merge_data
from data_split import split_data



def main():
    merge_data(100)
    
    dataset = utils.load_csv(
        environment.os.path.join(
            environment.DATA_PATH,
            'dataset.csv',
        )    
    )
    
    inputs, targets = split_data(dataset, 'experiment_1')
    
    k_fold = environment.sklearn.model_selection.StratifiedKFold(
        n_splits=10,
        shuffle=True,
    )
    
    for fold, (train_mask, validation_mask) in  enumerate(k_fold.split(inputs, targets)):
        inputs_train = inputs.iloc[train_mask]
        inputs_validation = inputs.iloc[validation_mask]
    
        targets_train = targets.iloc[train_mask]
        targets_validation = targets.iloc[validation_mask]
    
        for name, model in models_sklearn.models.items():
            print(environment.SEPARATOR_LINE)
            print(f'Model: {name}')
            train_sklearn.main_train(
                model(),
                inputs_train,
                targets_train,
                inputs_validation,
                targets_validation,
            )
        
    
if __name__ == "__main__":
    main()