### the previous python files cannot answer the questions so we do more experiments by adding more configs

### configs
# *** save_folder.239.T_50/p100/2f
# v4.mg.[0.1.2.3.30.31.60.61.62.73].p100.2f.T_1000 [those didn't have high acc so give them up]
# [???] v5.mg.73.p100.2f.T_50 [this config is also not good enough]
# [1] mg v7.mg.{0-89}.p100.2f.T_50

# *** save_folder.becat/p100/2f
# [2] mg v7.mg.{0-89}.p100.2f.T_50 
# [3] mg v6.mg.{0-89}.p100.2f.T_50
# [4] mg v6.mg.{0-89}.p100.2f.T_1000
# [5] mg v7.mg.{0-89}.p100.2f.T_1000

# [6] mg v8.mg.0.p100.2f.T_{10.20.30.40.50.100.200.500.1000}
# [7] gaussian v8.gaussian.{0.1;0.01;0.05;0.5;1;2;5;10;100;1000;10000}.p100.2f.T_{10.20.30.40.50.100.200.500.1000}
# [8] noise=0 v8.gaussian.{0}.p100.2f.T_{10.20.30.40.50.100.200}

### configs detail: [epsilon, distortion, clip, q, G_k, G_theta] from config file
# v6 batchsize int(10000*q); microbatches 1; q from config file
# v7 batchsize 64 microbatch 8; q=64/10000
# v8 batchsize 64 microbatch 8; q=64/10000


### answer questions
# 1. what is the relationship between (k, theta) - [tm, m, a]
# 2. try to find gaussian that reaches the smallest mia.
# 3. do we have same conclusions with different parameters? 
# [others] all MIA methods (RMIA,LIRA etc), datasets, task 




import plotly.graph_objects as go
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# import os
# all_dfs = []
# for filename in os.listdir("collect_results.T_50.T_1000"):
#     if filename.endswith(".csv"):
#         df = pd.read_csv(f"collect_results.T_50.T_1000/{filename}")
#         all_dfs.append(df)

# merged_df = pd.concat(all_dfs, ignore_index=True)
# merged_df.to_csv("collect_results.T_50.T_1000/collect_results.T_50.T_1000.csv", index=False)

# df = pd.read_csv("cfg_noise/collect_results.v7.old.csv")
# columns = ["eps", "distortion", "clip", "q", "G_k", "G_theta"]
# df = df[columns]
# from util import *
# cfg = preprocess_df(df)


# ### process experiments
# ## exp1
# # plrv | compare the machine results - [1] & [2]
# import matplotlib.pyplot as plt
# import seaborn as sns

# for filename in os.listdir("collect_results.becat"):
#     if filename.startswith("collect_results.1"):
#         df1 = pd.read_csv(f"collect_results.becat/{filename}")
#     if  filename.startswith("collect_results.2"):
#         df2 = pd.read_csv(f"collect_results.becat/{filename}")


# mia_239 = df1.mia.tolist()
# mia_bec = df2.mia.tolist()
# acc_239 = df1.acc.tolist()
# acc_bec = df2.acc.tolist()

# plt.figure(figsize=(10, 6))
# sns.histplot(mia_239, bins=30, kde=True, label='mia_239')
# sns.histplot(mia_bec, bins=30, kde=True, label='mia_becat')
# plt.title("Data Distribution")
# plt.xlabel("Values")
# plt.ylabel("Frequency")
# plt.legend()
# plt.savefig("plot_results.v7.3.exp1.mia.png")
# plt.cla()
# plt.clf()
# plt.figure(figsize=(10, 6))
# sns.histplot(acc_239, bins=30, kde=True, label='acc_239')
# sns.histplot(acc_bec, bins=30, kde=True, label='acc_bec')
# plt.title("Data Distribution")
# plt.xlabel("Values")
# plt.ylabel("Frequency")
# plt.legend()
# plt.savefig("plot_results.v7.3.exp1.acc.png")


# # ## exp2
# # plrv | T - [50, 1000]: a. [3] & [4] & [6](opt); b. [2] & [5] & [6](opt)
# df = pd.read_csv("cfg_noise/collect_results.v7.old.csv")
# columns = ["eps", "distortion", "clip", "q", "G_k", "G_theta"]
# df = df[columns]
# from util import *
# cfg = preprocess_df(df)

# df1 = "collect_results.becat.v6.T_50_1000.csv"
# df2 = "collect_results.becat.v7.T_50_1000.csv"
# df3 = "collect_results.becat.v8.csv"

# df1 = pd.read_csv(df1)
# df2 = pd.read_csv(df2)


# fig = go.Figure()

# # 添加 Theoretical MIA 数据
# fig.add_trace(go.Scatter3d(
#     x=df1.loc[df1['version']=='v6', 'G_k'],
#     y=df1.loc[df1['version']=='v6', 'G_theta'],
#     z=df1.loc[df1['version']=='v6', 'tmia'],
#     mode='markers',
#     marker=dict(size=5, color='blue'),
#     name='Theoretical MIA (bz:10000*q; mbz:1)'
# ))

# # 添加 Experimental MIA 数据
# fig.add_trace(go.Scatter3d(
#     x=df1.loc[df1['version']=='v6', 'G_k'],
#     y=df1.loc[df1['version']=='v6', 'G_theta'],
#     z=df1.loc[df1['version']=='v6', 'mia'],
#     mode='markers',
#     marker=dict(size=5, color='green'),
#     name='Experimental MIA (bz:10000*q; mbz:1)'
# ))

# # 添加 Accuracy 数据
# fig.add_trace(go.Scatter3d(
#     x=df1.loc[df1['version']=='v6', 'G_k'],
#     y=df1.loc[df1['version']=='v6', 'G_theta'],
#     z=df1.loc[df1['version']=='v6', 'acc'],
#     mode='markers',
#     marker=dict(size=5, color='red'),
#     name='Accuracy (bz:10000*q; mbz:1)'
# ))

# # 添加 Theoretical MIA 数据
# fig.add_trace(go.Scatter3d(
#     x=df2.loc[df2['version']=='v7', 'G_k'],
#     y=df2.loc[df2['version']=='v7', 'G_theta'],
#     z=df2.loc[df2['version']=='v7', 'tmia'],
#     mode='markers',
#     marker=dict(size=3, color='blue', symbol='x'),
#     name='Theoretical MIA (bz:64; mbz:8)'
# ))

# # 添加 Experimental MIA 数据
# fig.add_trace(go.Scatter3d(
#     x=df2.loc[df2['version']=='v7', 'G_k'],
#     y=df2.loc[df2['version']=='v7', 'G_theta'],
#     z=df2.loc[df2['version']=='v7', 'mia'],
#     mode='markers',
#     marker=dict(size=3, color='green', symbol='x'),
#     name='Experimental MIA (bz:64; mbz:8)'
# ))

# # 添加 Accuracy 数据
# fig.add_trace(go.Scatter3d(
#     x=df2.loc[df2['version']=='v7', 'G_k'],
#     y=df2.loc[df2['version']=='v7', 'G_theta'],
#     z=df2.loc[df2['version']=='v7', 'acc'],
#     mode='markers',
#     marker=dict(size=3, color='red', symbol='x'),
#     name='Accuracy (bz:64; mbz:8)'
# ))

# fig.write_html("plot_results.v7.3.exp2.v6.v7.html")



# ## exp3
df = pd.read_csv("cfg_noise/collect_results.v7.old.csv")
columns = ["eps", "distortion", "clip", "q", "G_k", "G_theta"]
df = df[columns]
from util import *
cfg = preprocess_df(df)

df1 = "collect_results.becat.v6.T_50_1000.csv"
df2 = "collect_results.becat.v7.T_50_1000.csv"
df3 = "collect_results.becat.v8.csv"

df1 = pd.read_csv(df1)
df2 = pd.read_csv(df2)
df = pd.read_csv(df3)

df['sigma'] = df['noise_type']  # 假设 noise_type 作为 sigma 使用

# 3D散点图：MIA
fig_mia = go.Figure(data=[
    go.Scatter3d(
        x=df['T'],
        y=df['sigma'],
        z=df['mia'],
        mode='markers',
        marker=dict(size=5, color=df['mia'], colorscale='Viridis', colorbar=dict(title='MIA')),
        name='MIA'
    )
])
fig_mia.update_layout(
    scene=dict(
        xaxis_title='T',
        yaxis_title='Sigma',
        zaxis_title='MIA'
    ),
    title="3D Scatter Plot of MIA by T and Sigma"
)
fig_mia.show()

# 3D散点图：ACC
fig_acc = go.Figure(data=[
    go.Scatter3d(
        x=df['T'],
        y=df['sigma'],
        z=df['acc'],
        mode='markers',
        marker=dict(size=5, color=df['acc'], colorscale='Plasma', colorbar=dict(title='ACC')),
        name='ACC'
    )
])
fig_acc.update_layout(
    scene=dict(
        xaxis_title='T',
        yaxis_title='Sigma',
        zaxis_title='ACC'
    ),
    title="3D Scatter Plot of ACC by T and Sigma"
)

fig.write_html("plot_results.v7.3.exp3.v8.html")


