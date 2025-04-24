# -*- coding: utf-8 -*- noqa
"""
Created on Sun Apr 20 23:11:45 2025

@author: Joel Tapia Salvador
"""
import environment

import metrics

def main_train(
        model,
        train_dataset, 
        validation_dataset,
):
    model.fit(train_dataset.inputs, train_dataset.targets)
    
    predictions = model.predict(validation_dataset.inputs)
    
    predictions = environment.torch.from_numpy(predictions)
    
    print(f'Accuracy: {metrics.accuracy(outputs=predictions, labels=validation_dataset.targets)}')
