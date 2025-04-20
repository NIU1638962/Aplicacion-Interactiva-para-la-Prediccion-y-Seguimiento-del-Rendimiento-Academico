# -*- coding: utf-8 -*- noqa
"""
Created on Sun Apr 20 19:52:37 2025

@author: Joel Tapia Salvador
"""
mapping = {
    "school": {
        "drop": True,    
    },
    "sex": {
        "drop": False,
        "new_name": "gender",
        "map_values": {
            "F": 0,
            "M": 1,
            None: -1,
        },
    },
    "age": {
         "drop": False,
         "new_name": "age",
         "map_values": lambda x: -1 if x is None else x,
    },
    "address": {
         "drop": False,
         "new_name": "area_residency",
         "map_values": {
            "U": 0,
            "R": 1,
            None: -1,
         },
    },
    "famsize": {
        "drop": False,
        "new_name": "size_family",
        "map_values": {
            "LE3": 2,
            "GT3": 4,
            None: -1,
         },
    },
    "Pstatus": {
        "drop": False,
        "new_name": "parental_marital_status",
        "map_values": {
            "T": 1,
            "A": 2,
            None: -1,
        }
    },
    "Medu": {
         "drop": False,
         "new_name": "mother_education",
         "map_values": {
              0: 0,
              1: 2,
              2: 3,
              3: 5,
              4: 11,
              None: -1,
         },
    },
    "Fedu": {
         "drop": False,
         "new_name": "father_education",
         "map_values": {
             0: 0,
             1: 2,
             2: 3,
             3: 5,
             4: 11,
             None: -1,
         },
    },
    "Mjob": {
        "drop": False,
        "new_name": "mother_occupation",
        "map_values": {
            "teacher": 3,
            "health": 3,
            "services": 3,
            "at_home": 2,
            "other": 6,
            None: -1,
        },
    },
    "Fjob": {
        "drop": False,
        "new_name": "father_occupation",
        "map_values": {
            "teacher": 3,
            "health": 3,
            "services": 3,
            "at_home": 2,
            "other": 6,
            None: -1,
        },
    },
    "reason": {
        "drop": True,   
    },
    "guardian": {
        "drop": False,
        "new_name": "guardian",
        "map_values": {
            "mother": 1,
            "father": 2,
            "other": 3,
            None: -1,
        },
    },
    "traveltime": {
        "drop": False,
        "new_name": "travel_time",
        "map_values": {
            1: 10,
            2: 22,
            3: 45,
            4: 120,
            None: -1,
        },
    },
    "studytime": {
         "drop": False,
         "new_name": "weekly_study_hours",
         "map_values": {
              1: 1.5,
              2: 4,
              3: 8,
              4: 12,
              None: -1,
         },
    },
    "failures": {
        "drop": False,
        "new_name": "failures_past_classes",
        "map_values": lambda x: -1 if x is None else x,
    },
    "schoolsup": {
         "drop": False,
         "new_name": "special_needs",
         "map_values": {
              "yes": 1,
              "no": 0,
              None: -1,
         },
    },
    "famsup": {
        "drop": False,
        "new_name": "additional_work",
        "map_values": {
             "yes": 1,
             "no": 0,
             None: -1,
        },
    },
    "paid": {
        "drop": False,
        "new_name": "personal_classes",
        "map_values": {
             "yes": 1,
             "no": 0,
             None: -1,
        },
    },
    "activities": {
         "drop": False,
         "new_name": "extra_curricular_activities",
         "map_values": {
              "yes": 1,
              "no": 0,
              None: -1,
         },
    },
    "nursery": {
        "drop": True,
    },
    "higher": {
        "drop": False,
        "new_name": "want_take_higher_education",
        "map_values": {
             "yes": 1,
             "no": 0,
             None: -1,
        },
    },
    "internet": {
         "drop": False,
         "new_name": "internet_access_home",
         "map_values": {
             "yes": 1,
             "no": 0,
             None: -1,
         },
    },
    "romantic": {
         "drop": False,
         "new_name": "marital_status",
         "map_values": {
             "yes": 5,
             "no": 4,
             None: -1,
         },
    },
    "famrel": {
         "drop": False,
         "new_name": "family_relationship",
         "map_values": lambda x: -1 if x is None else x - 1,
    },
    "freetime": {
         "drop": False,
         "new_name": "free_time",
         "map_values": lambda x: -1 if x is None else x - 1,
    },
    "goout": {
         "drop": False,
         "new_name": "go_out",
         "map_values": lambda x: -1 if x is None else x - 1,
    },
    "Dalc": {
         "drop": False,
         "new_name": "alcohol_workday",
         "map_values": lambda x: -1 if x is None else x - 1,
    },
    "Walc": {
         "drop": False,
         "new_name": "alcohol_weekend",
         "map_values": lambda x: -1 if x is None else x - 1,
    },
    "health": {
         "drop": False,
         "new_name": "health_status",
         "map_values": lambda x: -1 if x is None else x - 1,
    },
    "absences": {
         "drop": False,
         "new_name": "absences",
         "map_values": lambda x: -1 if x is None else x,
    },
    "G1": {
        "drop": True,    
    },
    "G2": {
         "drop": True,   
    },
    "G3": {
        "drop": False,
        "new_name": "target",
        "map_values": lambda x: -1 if x is None else x / 20,
    },
}