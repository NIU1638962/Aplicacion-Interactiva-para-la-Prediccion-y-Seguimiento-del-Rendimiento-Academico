# -*- coding: utf-8 -*- noqa
"""
Created on Sun Apr 20 16:40:30 2025

@author: Joel Tapia Salvador
"""
mapping = {
    "1": {
        "drop": False,
        "new_name": "age",
        "map_values": {
            1: 18,
            2: 22,
            3: 26,
            None: -1,
        },
    },
    "2": {
        "drop": False,
        "new_name": "gender",
        "map_values": {
            1: 0,
            2: 1,
            None: -1,
        },
    },
    "3": {
        "drop": False,
        "new_name": "previous_education_type",
        "map_values": {
            1: 0,
            2: 1,
            3: 2,
            None: -1,
        },
    },
    "4": {
        "drop": False,
        "new_name": "scholarship",
        "map_values": {
            1: 0,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            None: -1,
        },
    },
    "5": {
        "drop": False,
        "new_name": "additional_work",
        "map_values": {
            1: 1,
            2: 0,
            None: -1,
        },
    },
    "6": {
        "drop": False,
        "new_name": "extra_curricular_activities",
        "map_values": {
            1: 1,
            2: 0,
            None: -1,
        },
    },
    "7": {
        "drop": False,
        "new_name": "marital_status",
        "map_values": {
            1: 5,
            2: 4,
            None: -1,
        },
    },
    "8": {
        "drop": True,
    },
    "9": {
        "drop": False,
        "new_name": "transportation",
        "map_values": {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            None: -1,
        },
    },
    "10": {
        "drop": False,
        "new_name": "accomodation",
        "map_values": {
            1: 2,
            2: 1,
            3: 0,
            4: 3,
            None: -1,
        },
    },
    "11": {
        "drop": False,
        "new_name": "mother_education",
        "map_values": {
            1: 2,
            2: 3,
            3: 5,
            4: 11,
            5: 14,
            6: 17,
            None: -1,
        },
    },
    "12": {
        "drop": False,
        "new_name": "father_education",
        "map_values": {
            1: 2,
            2: 3,
            3: 5,
            4: 11,
            5: 14,
            6: 17,
            None: -1,
        },
    },
    "13": {
        "drop": False,
        "new_name": "number_siblings",
        "map_values": lambda x: -1 if x is None else x,
    },
    "14": {
        "drop": False,
        "new_name": "parental_marital_status",
        "map_values": lambda x: -1 if x is None else x,
    },
    "15": {
        "drop": False,
        "new_name": "mother_occupation",
        "map_values": lambda x: -1 if x is None else x,
    },
    "16": {
        "drop": False,
        "new_name": "father_occupation",
        "map_values": lambda x: -1 if x is None else x,
    },
    "17": {
        "drop": False,
        "new_name": "weekly_study_hours",
        "map_values": {
            1: 0,
            2: 3,
            3: 7,
            4: 16,
            5: 21,
            None: -1,
        },
    },
    "18": {
        "drop": False,
        "new_name": "reading_frequency_non_scientific",
        "map_values": {
            1: 0,
            2: 1,
            3: 2,
            None: -1,
        },
    },
    "19": {
        "drop": False,
        "new_name": "reading_frequency_scientific",
        "map_values": {
            1: 0,
            2: 1,
            3: 2,
            None: -1,
        },
    },
    "20": {
        "drop": False,
        "new_name": "attendance_seminars",
        "map_values": {
            1: 1,
            2: 0,
            None: -1,
        },
    },
    "21": {
        "drop": False,
        "new_name": "impact_projects_activities",
        "map_values": {
            1: 2,
            2: 0,
            3: 1,
            None: -1,
        },
    },
    "22": {
        "drop": False,
        "new_name": "attendance_classes",
        "map_values": {
            1: 2,
            2: 1,
            3: 0,
            None: -1,
        },
    },
    "23": {
        "drop": True,
    },
    "24": {
        "drop": True,
    },
    "25": {
         "drop": False,
         "new_name": "taking_notes_classes",
         "map_values": {
             1: 0,
             2: 1,
             3: 2,
             None: -1,
         },
    },
    "26": {
         "drop": False,
         "new_name": "listening_classes",
         "map_values": {
             1: 0,
             2: 1,
             3: 2,
             None: -1,
         },
    },
    "27": {
         "drop": False,
         "new_name": "discussion_improves_interest",
         "map_values": {
             1: 0,
             2: 1,
             3: 2,
             None: -1,
         },
    },
    "28": {
        "drop": True,    
    },
    "29": {
        "drop": True,
    },
    "30": {
        "drop": True,    
    },
    "COURSE ID": {
        "drop": True,
    },
    "GRADE": {
         "drop": False,
         "new_name": "target",
         "map_values": {
             0: 0,
             1: 0.25,
             2: 0.375,
             3: 0.5,
             4: 0.625,
             5: 0.75,
             6: 0.875,
             7: 1,
             None: -1,
         },
    },
    "STUDENT ID": {
        "drop": True,    
    },
}
