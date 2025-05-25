# -*- coding: utf-8 -*- noqa
"""
Created on Sun Apr 20 23:09:12 2025

@author: Joel Tapia Salvador
"""
import environment

models = {
    'decission_tree': environment.sklearn.tree.DecisionTreeClassifier,
    'random_forest': environment.sklearn.ensemble.RandomForestClassifier,
    'linear_regression': environment.sklearn.linear_model.LinearRegression,
}
