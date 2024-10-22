import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from noise.parser import parser_file
from noise.plrv import compute_obj_Gaussian
from noise.noise_params import distributions


alpha=0.05
sensitivities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

df = pd.DataFrame([])
for sensitivity in sensitivities:
    # header, content = parser_file(f"results/sen_{sensitivity}_alpha_{alpha}.txt")
    header, content = parser_file(f"results/14.sen_{sensitivity}_alpha_{alpha}_T_1.txt")
    dict_params = dict(zip(header, content))
    result_list = [{k: v for k, v in zip(header, values)} for values in content]
    result_list = pd.DataFrame(result_list)
    result_list['sensitivity'] = sensitivity
    result_list['mia'] = result_list[['mia']]
    
    df = pd.concat([df, pd.DataFrame(result_list)], ignore_index=True)

# EXP1: fix T; fix sensitivity; vary K; plot figure for both PLRV and Gaussian: [x]MIA -- [y]ACC_{best}
# Ks=[0.1, 0.2, 0.3, 0.4]
# acc = [[], []]
# for K in Ks:
#     filtered_df = df[(df['mia'] < K) & (df['mia'] > 0) & (df['sensitivity'] == 1.0)]
#     max_row = filtered_df.loc[filtered_df['obj'].idxmax()]
#     print("K:", K, "params:", dict(max_row))
#     obj_Gaussian, sigma_Gaussian = compute_obj_Gaussian(K, alpha, sensitivity)
#     print("K:", K, "sigma:", sigma_Gaussian)
#     acc[0].append(dict(max_row)['obj'])
#     acc[1].append(obj_Gaussian)
#     print("\n")

# plt.plot(Ks, acc[0], label="OurPLRV")
# plt.plot(Ks, acc[1], label="Gaussian")
# plt.title("Fix sensitivity Vary MIA advantage threshold")
# plt.xlabel('MIA threshold')
# plt.ylabel('Accuracy (mean of epsilon)')
# plt.grid()
# plt.savefig("results/sensitivisy=1.png")


# EXP2: [choice of noise with clipping] fix T; fix K; vary sensitivity; plot figure for both PLRV and Gaussian: [x]MIA -- [y]ACC_{best};
K=0.4
acc = [[], []]
for sensitivity in sensitivities:
    filtered_df = df[(df['sensitivity'] == sensitivity)]
    max_row = filtered_df.loc[filtered_df['obj'].idxmax()]
    print("sensitivity:", sensitivity, "params:", dict(max_row))
    obj_Gaussian, sigma_Gaussian = compute_obj_Gaussian(K, alpha, sensitivity)
    print("sensitivity:", sensitivity, "sigma:", sigma_Gaussian)
    acc[0].append(dict(max_row)['obj'])
    acc[1].append(obj_Gaussian)
    print("\n")

plt.plot(sensitivities, acc[0], label="OurPLRV")
plt.plot(sensitivities, acc[1], label="Gaussian")
plt.title("Fix MIA advantage threshold Vary sensitivity")
plt.xlabel('MIA threshold')
plt.ylabel('Accuracy (mean of epsilon)')
plt.grid()
plt.legend()
plt.savefig("results/14.k=0.4.png")


