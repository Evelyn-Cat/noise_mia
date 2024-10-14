import numpy as np

sensitivity = 1

max_order = 128
alphas = range(2, max_order + 1)

distributions = ["Exponential", "Uniform"]

noise_params = {
    0.3: {
        "G_k": 1.5,
        "G_theta": 0.1,
        "E_lambda": 0.5,
        "U_a": 0,
        "U_b": 1,
        "a1": 1,
        "a3": 0.5,
        "a4": 0.1
    }
}

search_range = {
    "a3": np.linspace(0.1, 0.9, 9),
    "a4": np.linspace(0.1, 0.9, 9),
    "E_lambda": [0.1, 0.5, 1, 5],  # E_lambda>0; t<E_lambda;
    ("U_b", "U_a"): [(1,0), (2,1)],  # b>a; when t=0: MGF=1;
}

# setted_delta = 1e-10
setted_delta = 1e-5
