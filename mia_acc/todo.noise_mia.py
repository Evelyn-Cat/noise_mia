import numpy as np
import pandas as pd
from itertools import product
from noise.noise_privacy import compute_mia, compute_obj
from noise.noise_params import orders, distributions, search_range

def main(sensitivity, alpha, T=1, dists="geu", filename=None, search_range=search_range):
    distributions = []
    if "g" in dists:
        distributions.append("Gamma")
    if "e" in dists:
        distributions.append("Exponential")
    if "u" in dists:
        distributions.append("Uniform")
    
    search_range = search_range[tuple(distributions)]
    
    expanded_items = []
    for key, values in search_range.items():
        expanded_items.append([value for value in values])
    all_combinations = list(product(*expanded_items))
    print(all_combinations[1])
    
    cnt = 0
    filename = f"results/{dists}.sen_{sensitivity}_alpha_{alpha}_T_{T}.txt" if filename==None else filename
    f = open(filename, "w", encoding="utf-8")
    for combination in all_combinations:
        param_dict = {}
        for i, key in enumerate(list(search_range.keys())):
            if isinstance(key, tuple):
                for sub_key, sub_value in zip(key, combination[i]):
                    param_dict[sub_key] = sub_value
            else:
                param_dict[key] = combination[i]
        
        if "U_b" in param_dict and "U_a" in param_dict:
            if param_dict['U_b'] <= param_dict['U_a']:
                continue
        
        betas, beta_index, beta, mia, epsilon, delta = compute_mia(param_dict, sensitivity=sensitivity, epsilon=param_dict['epsilon'], alpha=alpha, distributions=distributions, T=T)
        
        if betas and delta<1:
            obj1 = compute_obj(param_dict, distributions=distributions, mode=1)
            obj2 = compute_obj(param_dict, distributions=distributions, mode=2)
        
            if cnt == 0:
                if dists == "g":
                    line = "a1	    G_theta	G_k	    epsilon	mia	    delta	obj1	obj2\n"
                elif dists == "gu":
                    line = "a1	    a4	    G_theta	G_k	    U_b	    U_a	    epsilon	mia	    delta	obj1	obj2\n"
                elif dists == "geu":
                    line = "a1	    a3	    a4	    G_theta	G_k	    E_lmda  U_b	    U_a	    epsilon	mia	    delta	obj1	obj2\n"
                else:
                    raise NotImplementedError
                f.write(line)
                cnt = cnt + 1
            param_dict['mia'] = mia
            param_dict['epsilon'] = epsilon
            param_dict['delta'] = delta
            param_dict['obj1'] = obj1
            param_dict['obj2'] = obj2
            f.write("\t".join(f"{value:.3f}" for value in param_dict.values()) + '\n')
    f.close()


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
    T = int(sys.argv[3])
    distributions=sys.argv[4] # "geu"
    # print(sensitivity, alpha, T)
    main(sensitivity, alpha, T, distributions, search_range=search_range)

