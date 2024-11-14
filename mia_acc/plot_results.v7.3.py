### the previous python files cannot answer the questions so we do more experiments by adding more configs

### configs
# *** save_folder.239.T_50/p100/2f
# v4.mg.[0.1.2.3.30.31.60.61.62.73].p100.2f.T_1000 [those didn't have high acc so give them up]
# [???] v5.mg.73.p100.2f.T_50 [this config is also not good enough]
# [1] mg v7.mg.{0-89}.p100.2f.T_50

# *** save_folder.becat.T_50/p100/2f
# [2] mg v7.mg.{0-89}.p100.2f.T_50 

# *** save_folder.becat.T_1000/p100/2f
# [3] mg v6.mg.{0-89}.p100.2f.T_50
# [4] mg v6.mg.{0-89}.p100.2f.T_1000
# [5] mg v7.mg.{0-89}.p100.2f.T_1000
# [6] mg v8.mg.0.p100.2f.T_{10.20.30.40.50.100.200.500.1000}
# [7] gaussian v8.gaussian.{0.1;0.01;0.05;0.5}.p100.2f.T_{10.20.30.40.50.100.200.500.1000}
# [8] gaussian v8.gaussian.{0}.p100.2f.T_{10.20.30.40.50.100.200}
# [9] gaussian v8.gaussian.{1.2.5.10.100.1000.10000}.p100.2f.T_{10.20.30.40.50.100.200.500.1000}

### configs detail: [epsilon, distortion, clip, q, G_k, G_theta] from config file
# v6 batchsize int(10000*q); microbatches 1; q from config file
# v7 batchsize 64 microbatch 8; q from config file
# v8 batchsize 64 microbatch 8; q=64/10000

### process experiments
# plrv | compare the machine results - [1] & [2]
# plrv | T - [50, 1000]: a. [3] & [4] & [6](opt); b. [2] & [5] & [6](opt)
# plrv | 
# gaussian | T - [10.20.30.40.50.100.200.500.1000]
# gaussian | sigma - [0.1;0.01;0.05;0.5;1;2;5;10;100;1000;10000]
# gaussian | (T, sigma)
# (plrv, gaussian) -> optimal T


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

df = pd.read_csv("collect_results.v7.csv")

# 示例数据 (假设df包含 k, theta, tmia, mia, accuracy)
# 替换成实际数据
# df = pd.DataFrame({
#     'G_k': [10, 20, 30, 40],
#     'G_theta': [0.5, 1, 1.5, 2],
#     'tmia': [0.1, 0.15, 0.2, 0.25],
#     'mia': [0.05, 0.1, 0.15, 0.2],
#     'acc': [0.8, 0.85, 0.9, 0.95]
# })

# 创建图形对象
fig = go.Figure()

# 添加 Theoretical MIA 数据
fig.add_trace(go.Scatter3d(
    x=df['G_k'],
    y=df['G_theta'],
    z=df['tmia'],
    mode='markers',
    marker=dict(size=5, color='blue'),
    name='Theoretical MIA'
))

# 添加 Experimental MIA 数据
fig.add_trace(go.Scatter3d(
    x=df['G_k'],
    y=df['G_theta'],
    z=df['mia'],
    mode='markers',
    marker=dict(size=5, color='green'),
    name='Experimental MIA'
))

# 添加 Accuracy 数据
fig.add_trace(go.Scatter3d(
    x=df['G_k'],
    y=df['G_theta'],
    z=df['acc'],
    mode='markers',
    marker=dict(size=5, color='red'),
    name='acc'
))

# optimal one: 0.1#0.0064#2.05980287646780#1e-5#0.3097#9.999500033329732e-05
# mg: mia = 9.999500033329732e-05, acc = 0.3097, tmia = 
# gaussian: 100.0#0.4286#0.0040916179032535575
fig.add_trace(go.Scatter3d(
    x=[2.05980287646780],
    y=[1e-5],
    z=[9.999500033329732e-05],
    mode='markers',
    marker=dict(size=10, color='green', symbol='x'),
    name='opt_mia'
))
fig.add_trace(go.Scatter3d(
    x=[2.05980287646780],
    y=[1e-5],
    z=[0.3097],
    mode='markers',
    marker=dict(size=10, color='red', symbol='x'),
    name='opt_acc'
))


# 设置坐标轴和布局
fig.update_layout(
    title="3D Scatter Plot with Multiple Legends",
    scene=dict(
        xaxis_title='G_k',
        yaxis_title='G_theta',
        zaxis_title='Value'
    )
)

# 显示图形
fig.write_html("plot_results.v7.2.html")

