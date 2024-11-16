# v8 batchsize 64 microbatch 8; q=64/10000
import plotly.graph_objects as go
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# ## exp3


Ts = [10, 20, 30, 40, 50, 100, 200, 500, 1000]

sigmas = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 100, 1000, 10000]

df3 = "collect_results.becat.v8.T_1000.acc+1.csv"
df0 = pd.read_csv(df3)

df = df0.loc[df0['T']==float(1000), :]

mg_mia = 0.6993847718048646
mg_acc = 0.41041999999999995
print(mg_acc)


for T in Ts:
    df3 = f"collect_results.becat.v8.T_{T}.acc+1.csv"
    df0 = pd.read_csv(df3)
    
    df = df0.loc[df0['T']==float(T), :]
    
    mg_mia = df.loc[df['noise_type']==0.00,'mia'].mean()
    mg_acc = df.loc[df['noise_type']==0.00,'acc'].mean()

    
    results = [[], [], []]
    acc = []
    for sigma in sigmas:
        if T ==1000:
            dfi = df.loc[df['noise_type']==sigma,:]
            # dfi = df.loc[df['noise_type']==str(sigma),:]
        else:
            dfi = df.loc[df['noise_type']==sigma,:]
        print(dfi)
        
        mia_ab0 = np.array(dfi.loc[dfi['mia']>=0, 'mia'].tolist())
        mia_all = np.array(dfi.loc[:, 'mia'].tolist())
        mia_abs = np.array([np.abs(i) for i in dfi.loc[:, 'mia'].tolist()])
        print(len(dfi))

        mg = 'opt MG noise'
        # gaussian = [float(i) for i in sigma[1:]]
        
        mia_mg = mg_mia
        results[0].append(np.mean(mia_ab0))
        results[1].append(np.mean(mia_all))
        results[2].append(np.mean(mia_abs))
        acc.append(np.mean(np.array(dfi.loc[:, 'acc'].tolist())))
    
    # plt.plot([mg], [mia_mg], label='opt MG Noise', marker='o', markersize=8, color='blue', linestyle='None')
    # plt.axhline(y=mg_acc, label='opt MG Noise', marker='o', markersize=8, color='blue', linestyle='--')
    # plt.plot(sigmas, acc, label='Gaussian Noise', marker='x', markersize=6, color='orange', linestyle='-', linewidth=1)
    
    # plt.xscale('log')

    # plt.xlabel("Sigma")
    # plt.ylabel("ACC")
    # plt.title(f"T={T}")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f"becat.v8.T_{T}.acc.png")
    # plt.cla()
    # plt.clf()
    
    plt.axhline(y=mia_mg, label='opt MG Noise', marker='o', markersize=8, color='blue', linestyle='--')
    plt.plot(sigmas, results[0], label='Gaussian Noise', marker='x', markersize=6, color='orange', linestyle='-', linewidth=1)
    # plt.plot(sigmas, results[0], label='Gaussian Noise (mia>0)', marker='x', markersize=6, color='orange', linestyle='-', linewidth=1)
    # plt.plot(sigmas, results[1], label='Gaussian Noise (mia all)', marker='x', markersize=6, color='red', linestyle='-', linewidth=1)
    # plt.plot(sigmas, results[2], label='Gaussian Noise (mia abs)', marker='x', markersize=6, color='green', linestyle='-', linewidth=1)

    plt.xscale('log')

    plt.xlabel("Sigma")
    plt.ylabel("MIA")
    plt.title(f"T={T}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"becat.v8.T_{T}.png")
    plt.cla()
    plt.clf()