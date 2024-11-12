import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib import cm
import pandas as pd
import numpy as np

df = pd.read_csv("collect_results.v7.csv")

# 假设数据适合进行插值，可以创建网格
k = df['G_k'].values
theta = df['G_theta'].values

# 创建网格
K, Theta = np.meshgrid(np.linspace(k.min(), k.max(), 50), np.linspace(theta.min(), theta.max(), 50))
fig = plt.figure(figsize=(15, 5))

# 曲面图 for `theoritical_mia`
ax1 = fig.add_subplot(131, projection='3d')
Z1 = griddata((k, theta), df['tmia'], (K, Theta), method='cubic')
surf1 = ax1.plot_surface(K, Theta, Z1, cmap='cividis', edgecolor='none')
ax1.set_xlabel('k')
ax1.set_ylabel('theta')
ax1.set_zlabel('theoritical_mia')
ax1.set_title('Theoritical MIA')
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# 曲面图 for `theoritical_mia`
ax2 = fig.add_subplot(132, projection='3d')
Z2 = griddata((k, theta), df['mia'], (K, Theta), method='cubic')
surf2 = ax2.plot_surface(K, Theta, Z2, cmap='cividis', edgecolor='none')
ax2.set_xlabel('k')
ax2.set_ylabel('theta')
ax2.set_zlabel('Experimental MIA')
ax2.set_title('Experimental MIA')
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

# 曲面图 for `theoritical_mia`
ax3 = fig.add_subplot(133, projection='3d')
Z3 = griddata((k, theta), df['acc'], (K, Theta), method='cubic')
surf3 = ax3.plot_surface(K, Theta, Z3, cmap='cividis', edgecolor='none') # cmap=cm.viridis
ax3.set_xlabel('k')
ax3.set_ylabel('theta')
ax3.set_zlabel('Accuracy')
ax3.set_title('Accuracy')
fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)


# plt.tight_layout()
fig.savefig("plot_results.v7.1.png")