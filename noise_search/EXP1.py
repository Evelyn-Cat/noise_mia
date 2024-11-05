import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from noise.parser import parser_file
from noise.plrv import compute_obj_Gaussian


dists = "g" # g gu geu
alpha = 0.15 # 0.15 0.2
sensitivities = [1.0] # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # 2.0, 3.0, 4.0, 5.0

df = pd.DataFrame([])
for sensitivity in sensitivities:
    # header, content = parser_file(f"results/sen_{sensitivity}_alpha_{alpha}.txt")
    filename = f"results/{dists}.sen_{sensitivity}_alpha_{alpha}_T_1.txt"
    content = parser_file(filename)
    
    if dists == "g":
        header = ["a1", "G_theta", "G_k", "epsilon", "mia", "delta", "obj1", "obj2"]
    if dists == "gu":
        header = []
    if dists == "geu":
        header = ["a1", "a3", "a4", "G_theta", "G_k", "E_lambda", "U_b", "U_a", "epsilon", "mia", "delta", "obj1", "obj2"]
    
    result_list = [{k: v for k, v in zip(header, values)} for values in content]
    result_list = pd.DataFrame(result_list)
    result_list['sensitivity'] = sensitivity
    result_list['mia'] = result_list[['mia']]

    df = pd.concat([df, pd.DataFrame(result_list)], ignore_index=True)

# EXP1: fix T; fix sensitivity; vary K; plot figure for both PLRV and Gaussian: [x]MIA -- [y]ACC_{best}
# Ks=np.linspace(0.0003, 0.09957, 5)
Ks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
acc = [[], []]
for K in Ks:
    filtered_df = df[(df['mia'] < K) & (df['sensitivity'] == 1.0)]
    max_row = filtered_df.loc[filtered_df['obj1'].idxmax()]
    print("K:", K, "params:", dict(max_row))
    obj_Gaussian, sigma_Gaussian = compute_obj_Gaussian(K, alpha, sensitivity)
    print("K:", K, "sigma:", sigma_Gaussian)
    acc[0].append(dict(max_row)['obj1'])
    acc[1].append(obj_Gaussian)
    print("\n")
    

plt.plot(Ks, acc[0], label="OurPLRV")
plt.plot(Ks, acc[1], label="Gaussian")
plt.title("Fix sensitivity Vary MIA advantage threshold")
plt.xlabel('MIA threshold')
plt.ylabel('Accuracy (mean of epsilon)')
plt.grid()
plt.legend()
plt.savefig(f"results/{dists}.exp1.alpha={alpha}.sen=1.png")

