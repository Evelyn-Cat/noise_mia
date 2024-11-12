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

