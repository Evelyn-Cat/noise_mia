import numpy as np
from itertools import product
from noise.noise_privacy_2 import compute_rdp_alpha, compute_mia
from noise.noise_params_2 import alphas, distributions, search_range

expanded_items = []
for key, values in search_range.items():
    expanded_items.append([value for value in values])
all_combinations = list(product(*expanded_items))
print(all_combinations[1])

def main(epsilon, sensitivity):
    param_dict = {}
    for combination in all_combinations:
        for i, key in enumerate(list(search_range.keys())):
            if isinstance(key, tuple):
                for sub_key, sub_value in zip(key, combination[i]):
                    param_dict[sub_key] = sub_value
            else:
                param_dict[key] = combination[i]
        
        rdp_Ns = []
        betas, beta_index, beta, mia = compute_mia(param_dict, sensitivity, epsilon=epsilon)
        if betas:
            print(beta_index)
        
        print(param_dict)
        print(rdp_Ns)
        



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
    # alpha = 0.3
    # sensitivity = 1
    # rdp_N_ = compute_rdp_alpha(N, alpha, sensitivity)
    # print(f"RDP of noise (alpha={alpha}, sensitivity={sensitivity}) = {rdp_N_}")

    import sys
    epsilon = sys.argv[1]
    sensitivity = sys.argv[2]
    print(epsilon, sensitivity)
    main(epsilon, sensitivity)


