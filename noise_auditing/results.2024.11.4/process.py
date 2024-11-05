import os
import pandas as pd
from itertools import product

folders = ["logs.2024.11.3", "logs.logs.new", "logs.logs.new1"]
col = ["dataset", "model", "noise_type", "noise_param", "epoch", "l2_norm_clip", "acc"]

cnt = 0
df = pd.DataFrame([], columns = col)
for folder in folders:
    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)
        with open(filepath, 'r', encoding='utf-8') as f:
            rs = f.readlines()
            for r in rs:
                if "print_results" in r:
                    exp_name, acc = r.strip().split("print_results: ")[1].split(" ")
                    dataset, model, noise_type, noise_params, epochs, l2_norm_clip = exp_name.split("-")
                    acc = float(acc)
                    
                    df.loc[cnt, col] = [dataset, model, noise_type, noise_params, epochs, l2_norm_clip, acc]
                    cnt = cnt + 1
# print(df)

## colormap 1 of K, Theta - MIA
## colormpa 2 of K, Theta - ACC



datasets = list(set(df['dataset'].tolist()))
models = list(set(df['model'].tolist()))
noise_params = list(set(df['noise_param'].tolist()))
epochs = list(set(df['epoch'].tolist()))
l2_norm_clips = list(set(df['l2_norm_clip'].tolist()))
datasets = ["p100"]
models = ["lr"]
print(noise_params)
print(epochs)
print(l2_norm_clips)

results = {(i, j, k):[] for i, j, k in product(datasets, models, ["gaussian", "lmo"])}
for dataset, model, noise_param, epoch, l2_norm_clip in product(datasets, models, noise_params, epochs, l2_norm_clips):
    filtered_df = df[
        (df['dataset'] == dataset) 
        & (df['model'] == model)
        & (df['noise_param'] == noise_param)
        & (df['epoch'] == epoch)
        & (df['l2_norm_clip'] == l2_norm_clip)
    ]
    
    if len(filtered_df) < 2:
        continue
    else:
        print(filtered_df)
        acc_gaussian = filtered_df.loc[filtered_df["noise_type"]=="gaussian", "acc"].values
        print("acc_gaussian", acc_gaussian)
        acc_lmo = filtered_df.loc[filtered_df["noise_type"]=="lmo", "acc"].values
        print("acc_lmo", acc_lmo)
        results[(dataset, model, "gaussian")].append(acc_gaussian)
        results[(dataset, model, "lmo")].append(acc_lmo)
        # print(results)
        

# print(results)


# import matplotlib.pyplot as plt
# import numpy as np

# # Create a gradient
# gradient = np.linspace(0, 1, 256)
# gradient = np.vstack((gradient, gradient))

# # Plot
# plt.imshow(gradient, aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('Viridis Colormap')
# plt.show()
