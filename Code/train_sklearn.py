# -*- coding: utf-8 -*- noqa
"""
Created on Sun Apr 20 23:11:45 2025

@author: Joel Tapia Salvador
"""
import environment

import metrics

def main_train(
        model,
        inputs_train, 
        targets_train, 
        inputs_validation, 
        targets_validation,
):
    model.fit(inputs_train, targets_train)
    
    predictions = model.predict(inputs_validation)
    
    targets_validation = environment.torch.from_numpy(
        targets_validation.to_numpy()
    )
    
    predictions = environment.torch.from_numpy(predictions)
    
    print(f'Accuracy: {metrics.accuracy(outputs=predictions, labels=targets_validation)}')
