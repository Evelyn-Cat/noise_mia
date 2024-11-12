# answer the questions
# 1. what is the relationship between (k, theta) - [tm, m, a]
# 2. try to find gaussian that reaches the smallest mia.
# 3. do we have same conclusions with different parameters? 
# [others] all MIA methods (RMIA,LIRA etc), datasets, task 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

df = pd.read_csv("collect_results.v7.csv")

fig = plt.figure(figsize=(15, 5))

# 3D scatter plot for `theoritical_mia`
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(df['G_k'], df['G_theta'], df['tmia'], c='b', label='Theoritical MIA')
ax1.set_xlabel('k')
ax1.set_ylabel('theta')
ax1.set_zlabel('Theoritical MIA')
ax1.set_title('Theoritical MIA')

# 3D scatter plot for `mia`
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(df['G_k'], df['G_theta'], df['mia'], c='g', label='Experimental MIA')
ax2.set_xlabel('k')
ax2.set_ylabel('theta')
ax2.set_zlabel('Experimental MIA')
ax2.set_title('Experimental MIA')

# 3D scatter plot for `accuracy`
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(df['G_k'], df['G_theta'], df['acc'], c='r', label='Accuracy')
ax3.set_xlabel('k')
ax3.set_ylabel('theta')
ax3.set_zlabel('Accuracy')
ax3.set_title('Accuracy')

# plt.tight_layout()
plt.savefig("collect_results.v7.png")


