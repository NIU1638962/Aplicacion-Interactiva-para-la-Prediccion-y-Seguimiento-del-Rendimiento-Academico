# -*- coding: utf-8 -*- noqa
"""
Created on Sun Apr 20 21:08:59 2025

@author: Joel Tapia Salvador
"""
import environment
import utils

def split_data(dataframe, mode, seed=None):
    columns = dataframe.columns.drop(['target'])
    inputs = dataframe[columns]
    targets = dataframe['target'].astype('int64')
    
    if mode == "experiment_1":
        k_fold = environment.sklearn.model_selection.StratifiedKFold(
            n_splits=20,
            shuffle=True,
            random_state=seed,
        )
        _, mask = next(k_fold.split(inputs, targets))
        
        del _
        utils.collect_memory()
        
        inputs = inputs.iloc[mask]
        targets = targets.iloc[mask]
        
        return inputs, targets
        
        
        
        
    
    
    
    
    