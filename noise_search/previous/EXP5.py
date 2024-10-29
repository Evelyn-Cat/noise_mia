import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from noise.parser import parser_file
from noise.plrv import compute_obj_Gaussian
from noise_mia.noise.noise_params_134 import distributions, orders
from noise_mia.noise.noise_privacy_134 import compute_rdp_order
from noise.rdp_accounting import get_privacy_spent


alpha=0.2
sensitivity=1.0
Ts=[1, 5, 10, 20, 50, 100]

df = pd.DataFrame([])
for T in Ts:
    header, content = parser_file(f"results/sen_{sensitivity}_alpha_{alpha}_T_{T}.txt")
    dict_params = dict(zip(header, content))
    result_list = [{k: v for k, v in zip(header, values)} for values in content]
    result_list = pd.DataFrame(result_list)
    result_list['sensitivity'] = sensitivity
    result_list['T'] = T
    result_list['mia'] = result_list[['mia']]
    
    df = pd.concat([df, pd.DataFrame(result_list)], ignore_index=True)


# EXP5: vary T
K=0.2

acc = [[], []]
for T in Ts:
    filtered_df = df[(df['T'] == T)]
    max_row = filtered_df.loc[filtered_df['obj'].idxmax()]
    print("T:", T, "params:", dict(max_row))
    obj_Gaussian, sigma_Gaussian = compute_obj_Gaussian(K, alpha, sensitivity)
    print("T:", T, "sigma:", sigma_Gaussian)
    acc[0].append(dict(max_row)['obj'])
    acc[1].append(obj_Gaussian)
    print("\n")


plt.plot(Ts, acc[0], label="OurPLRV")
plt.plot(Ts, acc[1], label="Gaussian")
plt.title("Fix MIA advantage threshold Fix sensitivity and Vary T")
plt.xlabel('MIA threshold')
plt.ylabel('Accuracy (mean of epsilon)')
plt.grid()
plt.legend()
plt.savefig("results/exp5.alpha=0.2.sen=1k=0.2varyT.png")

