# -*- coding: utf-8 -*- noqa
"""
Created on Mon Jun 23 04:54:04 2025

@author: JoelT
"""
import environment


class Dataset(environment.torch.utils.data.Dataset):
    __slots__ = ('__feature_names', '__inputs', '__targets')

    def __init__(self, inputs, targets, feature_names):
        self.__inputs = inputs
        self.__targets = targets
        self.__feature_names = feature_names

    def __getitem__(self, index):
        return self.__inputs[index], self.__targets[index]

    def __len__(self):
        return len(self.__inputs)

    @property
    def feature_names(self):
        return self.__feature_names

    @property
    def inputs(self):
        return self.__inputs

    @property
    def targets(self):
        return self.__targets
