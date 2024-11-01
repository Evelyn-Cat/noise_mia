import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from noise.parser import parser_file
from noise.plrv import compute_obj_Gaussian


dists = "geu"  # gu geu
alpha=0.15

sensitivity=1.0
Ts=[1, 5, 10, 20, 50, 100]

df = pd.DataFrame([])
for T in Ts:
    content = parser_file(f"results/{dists}.sen_{sensitivity}_alpha_{alpha}_T_{T}.txt")
    
    if dists == "g":
        header = ["a1", "G_theta", "G_k", "epsilon", "mia", "delta", "obj1", "obj2"]
    if dists == "gu":
        header = []
    if dists == "geu":
        header = ["a1", "a3", "a4", "G_theta", "G_k", "E_lambda", "U_b", "U_a", "epsilon", "mia", "delta", "obj1", "obj2"]
    
    result_list = [{k: v for k, v in zip(header, values)} for values in content]
    result_list = pd.DataFrame(result_list)
    result_list['sensitivity'] = sensitivity
    result_list['T'] = T
    result_list['mia'] = result_list[['mia']]
    
    df = pd.concat([df, pd.DataFrame(result_list)], ignore_index=True)


# EXP5: vary T
K=0.005

acc = [[], []]
for T in Ts:
    filtered_df = df[(df['T'] == T)]
    max_row = filtered_df.loc[filtered_df['obj1'].idxmax()]
    print("T:", T, "params:", dict(max_row))
    obj_Gaussian, sigma_Gaussian = compute_obj_Gaussian(K, alpha, sensitivity)
    print("T:", T, "sigma:", sigma_Gaussian)
    acc[0].append(dict(max_row)['obj1'])
    acc[1].append(obj_Gaussian)
    print("\n")


plt.plot(Ts, acc[0], label="OurPLRV")
plt.plot(Ts, acc[1], label="Gaussian")
plt.title("Fix MIA advantage threshold Fix sensitivity and Vary T")
plt.xlabel('MIA threshold')
plt.ylabel('Accuracy (mean of epsilon)')
plt.grid()
plt.legend()
plt.savefig(f"results/{dists}.exp5.varyT.alpha={alpha}.sen=1.k=0.005.png")

