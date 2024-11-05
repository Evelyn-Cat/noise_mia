import numpy as np

# sensitivity = 1
# alpha = 0.05
# K = 0.2

max_order = 128
orders = range(2, max_order + 1)

distributions = ["Gamma", "Exponential", "Uniform"]
threshold = {
    "Gamma": 0.1,
    "Exponential": 0.1,
    "Uniform": 0.1
}


# noise_params = {
#     0.3: {
#         "G_k": 1.5,
#         "G_theta": 0.1,
#         "E_lambda": 0.5,
#         "U_a": 0,
#         "U_b": 1,
#         "a1": 1,
#         "a3": 0.5,
#         "a4": 0.1
#     },
#     2.402: {
#         'a1': 0.1, 
#         'a3': 0.1, 
#         'a4': 0.1, 
#         'G_theta': 7.5, 
#         'G_k': 1.0, 
#         'E_lambda': 0.1, 
#         'U_b': 2.0, 
#         'U_a': 1.0, 
#         'epsilon': 2.402, 
#         'mia': 0.0022600000000000398, 
#         'obj': 19.0, 
#         'sensitivity': 1.0,
#         "gaussian": 0.49256859462010105
#     },
#     0.01: {'a1': 0.5, 'a3': 0.9, 'a4': 0.1, 'G_theta': 7.5, 'G_k': 1.0, 'E_lambda': 0.1, 'U_b': 1.0, 'U_a': 0.0, 'epsilon': 0.01, 'mia': 0.00029, 'obj': 18.0, 'delta': 0.00716, 'sensitivity': 1.0},
#     0.10967: {'a1': 0.1, 'a3': 0.1, 'a4': 0.1, 'G_theta': 7.5, 'G_k': 1.0, 'E_lambda': 0.1, 'U_b': 2.0, 'U_a': 1.0, 'epsilon': 0.10967, 'mia': 0.00482, 'obj': 19.0, 'delta': 0.3416, 'sensitivity': 1.0}
# }

noise_params = {
    # 2024.10.31 sensitivity: 1-5; K: 0.005; from EXP1
    ("g", 0.2, 1): {'a1': 1.0, 'G_theta': 10.0, 'G_k': 10.0, 'epsilon': 0.01, 'mia': 0.001, 'delta': 0.0, 'obj1': 100.0, 'obj2': 0.0, 'sensitivity': 1.0, 'gaussian': 0.7243976515506757},
    ("g", 0.15, 1):{'a1': 1.0, 'G_theta': 10.0, 'G_k': 10.0, 'epsilon': 0.01, 'mia': 0.001, 'delta': 0.0, 'obj1': 100.0, 'obj2': 0.0, 'sensitivity': 1.0, 'gaussian': 0.6967240601856282},
    ("g", 0.2, 2): {'a1': 1.0, 'G_theta': 10.0, 'G_k': 10.0, 'epsilon': 0.01, 'mia': 0.001, 'delta': 0.0, 'obj1': 100.0, 'obj2': 0.0, 'sensitivity': 2.0, 'gaussian': 1.4487953031013514},
    ("g", 0.2, 3): {'a1': 1.0, 'G_theta': 10.0, 'G_k': 10.0, 'epsilon': 0.01, 'mia': 0.001, 'delta': 0.0, 'obj1': 100.0, 'obj2': 0.0, 'sensitivity': 3.0, 'gaussian': 2.173192954652027},
    ("g", 0.2, 4): {'a1': 1.0, 'G_theta': 10.0, 'G_k': 10.0, 'epsilon': 0.01, 'mia': 0.001, 'delta': 0.0, 'obj1': 100.0, 'obj2': 0.0, 'sensitivity': 4.0, 'gaussian': 2.897590606202703},
    ("g", 0.2, 5): {'a1': 1.0, 'G_theta': 10.0, 'G_k': 10.0, 'epsilon': 0.01, 'mia': 0.001, 'delta': 0.0, 'obj1': 100.0, 'obj2': 0.0, 'sensitivity': 5.0, 'gaussian': 3.6219882577533786},
    ("geu", 0.2, 1): {'a1': 0.1, 'a3': 0.1, 'a4': 0.1, 'G_theta': 3.0, 'G_k': 2.0, 'E_lambda': 0.1, 'U_b': 1.0, 'U_a': 0.0, 'epsilon': 0.01, 'mia': 0.001, 'delta': 0.0, 'obj1': 6.0, 'obj2': 0.001, 'sensitivity': 1.0, 'gaussian': 0.7243976515506757},
    ("geu", 0.15, 1): {'a1': 0.1, 'a3': 0.1, 'a4': 0.1, 'G_theta': 3.0, 'G_k': 2.0, 'E_lambda': 0.1, 'U_b': 1.0, 'U_a': 0.0, 'epsilon': 0.01, 'mia': 0.001, 'delta': 0.0, 'obj1': 6.0, 'obj2': 0.001, 'sensitivity': 1.0, 'gaussian': 0.6967240601856282}
}

# search_range = {
#     "a1": np.linspace(0.1, 0.9, 9),
#     "a3": np.linspace(0.1, 0.9, 9),
#     "a4": np.linspace(0.1, 0.9, 9),
#     ("G_theta", "G_k"): [(1,2), (2,2), (3,2), (5,1), (9,0.5), (7.5,1), (0.5,1)],  # k>0; theta>0; t<1/theta
#     "E_lambda": [0.1, 0.5, 1, 5],  # E_lambda>0; t<E_lambda;
#     ("U_b", "U_a"): [(1,0), (2,1)],  # b>a; when t=0: MGF=1;
#     "epsilon": np.linspace(0.01, 3, 31),
# }

search_range ={
    ("Gamma", ): {
        "a1": [1],
        "G_theta": np.linspace(0.1, 10, 10), # theta>0; t<1/theta
        "G_k": np.linspace(0.1, 10, 10), # k>0; t<1/theta
        "epsilon": np.linspace(0.01, 3, 31),
    },
    ("Gamma", "Uniform"): {
        "a1": np.linspace(0.1, 0.9, 3),
        "a4": np.linspace(0.1, 0.9, 3),
        "G_theta": np.linspace(0.1, 10, 10), # theta>0; t<1/theta
        "G_k": np.linspace(0.1, 10, 10), # k>0; t<1/theta
        "U_b": np.linspace(1, 10, 10), # b>a; when t=0: MGF=1;
        "U_a": np.linspace(1, 10, 10), # b>a; when t=0: MGF=1;
        "epsilon": np.linspace(0.01, 3, 31),
    },
    ("Gamma", "Exponential", "Uniform"): {
        "a1": np.linspace(0.1, 0.9, 3),
        "a3": np.linspace(0.1, 0.9, 3),
        "a4": np.linspace(0.1, 0.9, 3),
        ("G_theta", "G_k"): [(1,2), (2,2), (3,2), (5,1), (9,0.5), (7.5,1), (0.5,1)],  # k>0; theta>0; t<1/theta
        "E_lambda": [0.1, 0.5, 1, 5],  # E_lambda>0; t<E_lambda;
        ("U_b", "U_a"): [(1,0), (2,1)],  # b>a; when t=0: MGF=1;
        "epsilon": np.linspace(0.01, 3, 31),
    }
}