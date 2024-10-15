import numpy as np
import pandas as pd
from itertools import product
from noise.noise_privacy import compute_rdp_order, compute_mia
from noise.noise_params import orders, distributions, search_range

expanded_items = []
for key, values in search_range.items():
    expanded_items.append([value for value in values])
all_combinations = list(product(*expanded_items))
print(all_combinations[1])

def main(sensitivity, alpha):
    df = pd.DataFrame([])
    for combination in all_combinations:
        param_dict = {}
        for i, key in enumerate(list(search_range.keys())):
            if isinstance(key, tuple):
                for sub_key, sub_value in zip(key, combination[i]):
                    param_dict[sub_key] = sub_value
            else:
                param_dict[key] = combination[i]
        
        betas, beta_index, beta, mia = compute_mia(param_dict, sensitivity, param_dict['epsilon'], alpha=alpha)
        objective = param_dict["G_k"] * param_dict["G_theta"] + (param_dict['U_a'] + param_dict['U_b'])/2 + 1/param_dict['E_lambda']
        
        if betas:
            param_dict['mia'] = mia
            param_dict['obj'] = objective
            df = df.append(param_dict, ignore_index=True)
    
    return df


if __name__ == '__main__':
    # N = {
    #     "G_k": 1.5,
    #     "G_theta": 0.1,
    #     "E_lambda": 0.5,
    #     "U_a": 0,
    #     "U_b": 1,
    #     "a1": 1,
    #     "a3": 0.5,
    #     "a4": 0.1
    # }
    # order = 0.3
    # sensitivity = 1
    # rdp_N_ = compute_rdp_order(N, order, sensitivity)
    # print(f"RDP of noise (order={order}, sensitivity={sensitivity}) = {rdp_N_}")

    import sys
    sensitivity = float(sys.argv[1])
    alpha = float(sys.argv[2])
    df = main(sensitivity, alpha)
    df.to_csv(f"results/sen_{sensitivity}_alpha_{alpha}.csv")

