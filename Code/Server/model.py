# -*- coding: utf-8 -*- noqa
"""
Created on Thu Jun 26 05:43:44 2025

@author: JoelT
"""
import environment


MODEL = environment.sklearn.ensemble.RandomForestClassifier

PARAMETRES = {
    'n_estimators': 300,
    'criterion': 'gini',
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'min_weight_fraction_leaf': 0.0,
    'max_features': 'sqrt',
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'bootstrap': True,
    'oob_score': False,
    'n_jobs': None,
    'random_state': environment.SEED,
    'verbose': 0,
    'warm_start': False,
    'class_weight': None,
    'ccp_alpha': 0.0,
    'max_samples': None,
}
