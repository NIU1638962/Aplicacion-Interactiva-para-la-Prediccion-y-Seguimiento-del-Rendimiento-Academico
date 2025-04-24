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
from datasets import get_datasets, k_fold



def main():
    utils.print_message(
        environment.SECTION_LINE
        + '\nSTARTING MAIN PROGRAM'
        
    )
    environment.NUMBER_GRADES = 100
    
    merge_data()
    
    dataset = get_datasets('experiment_1')
    
    utils.print_message(
        environment.SECTION_LINE
        + '\nSTARTING TRAINING'
        
    )
    for name, model in models_sklearn.models.items():
        utils.print_message(
            f'{environment.SEPARATOR_LINE}\nTraining Model: {name}'
        )
        
        for fold, train_dataset, validation_dataset in k_fold(dataset, 10):
            utils.print_message(
                f'{environment.SECTION * 3}'
                + f'Fold: {fold}'
                + f'{environment.SECTION * 3}'
            )
            
            train_sklearn.main_train(
                model(),
                train_dataset,
                validation_dataset,
            )
        
    
if __name__ == "__main__":
    main()