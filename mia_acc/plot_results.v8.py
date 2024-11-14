# v8 batchsize 64 microbatch 8; q=64/10000
import plotly.graph_objects as go
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# ## exp3
df3 = "collect_results.becat.v8.csv"
df = pd.read_csv(df3)
print(df.head())

Ts = [10, 20, 30, 40, 50, 100, 200, 500, 1000]
sigma = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 100, 1000, 10000]

for T in Ts:
    sigma = df.loc[df['T']==float(T), 'noise_type'].tolist()
    mia = df.loc[df['T']==float(T), 'mia'].tolist()

    mg = 'opt mg noise'
    gaussian = [float(i) for i in sigma[1:]]
    
    mia_mg = mia[0]
    mia_gaussian = mia[1:]
    
    # plt.plot([mg], [mia_mg], label='opt MG Noise', marker='o', markersize=8, color='blue', linestyle='None')
    plt.axhline(y=mia_mg, label='opt MG Noise', marker='o', markersize=8, color='blue', linestyle='--')
    plt.plot(gaussian, mia_gaussian, label='Gaussian Noise', marker='x', markersize=6, color='orange', linestyle='-', linewidth=1)
    
    plt.xscale('log')

    plt.xlabel("Sigma")
    plt.ylabel("MIA")
    plt.title(f"T={T}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"becat.v8.T_{T}.png")
    plt.cla()
    plt.clf()