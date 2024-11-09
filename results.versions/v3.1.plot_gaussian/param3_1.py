import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# import pandas as pd
# from param import config_acc, config_mia
# configs = pd.read_excel("../Qt.xlsx")

# mia_lmo, acc_lmo, epsilon, sigma, q, clip, k, theta = [], [], [], [], [], [], [], []
# for i in range(90):
#     if i in config_mia and i in config_acc:
#         # print(config_mia[i], config_acc[i])
#         mia_lmo.append(config_mia[i])
#         acc_lmo.append(config_acc[i] * 100)
        
#         epsilon.append(configs.loc[i, "epsilon"])
#         sigma.append(configs.loc[i, "sigma"])
#         clip.append(configs.loc[i, "clip"])
#         q.append(configs.loc[i, "q"])
#         k.append(configs.loc[i, "k"])
#         theta.append(configs.loc[i, "theta"])

# print(mia_lmo)
# print(acc_lmo)      
# print(epsilon)
# print(sigma)
# print(clip)
# print(q)
# print(k)
# print(theta)
# print('\n')


mia_lmo=[0.08434114843375096, 0.08231704248921375, 0.062880898039201, 0.08810244822154573, 0.0769610411361284, 0.08369755902156667, 0.07436508195530674, 0.0822249402556695, 0.07166928669035968, 0.06849941642469226, 0.07027205298533169, 0.07176236622689079, 0.07991962317587344, 0.07705362944229147, 0.08010424424166136, 0.0829615207143473, 0.07352923328811553, 0.07807153542015544, 0.07334339422420959, 0.07455073126429625, 0.07529298403543135, 0.07408654335269817, 0.06859279146561167, 0.07436508195530674, 0.06812582904760044, 0.07631268284916351, 0.07529298403543135, 0.07547846117590556, 0.07510747248680548, 0.0822249402556695, 0.07455073126429625, 0.07677583880204963, 0.0764053312024053, 0.0693394780786129, 0.0701788346212465, 0.07760897932698288, 0.07436508195530674, 0.07705362944229147, 0.0820407103362094, 0.08010424424166136, 0.06896620446472282, 0.08194858264716733, 0.08038111194681256, 0.06961934188005707, 0.08837711061741292, 0.08148781684626792]
acc_lmo=[73.5, 73.66, 71.12, 73.19, 72.50999999999999, 72.89999999999999, 73.35000000000001, 72.33000000000001, 73.02, 71.69, 72.72, 72.87, 74.18, 72.19, 73.41, 73.09, 72.85000000000001, 72.18, 73.11999999999999, 72.11, 72.65, 73.0, 72.77, 72.97, 72.86, 73.94, 72.16, 73.02, 72.92999999999999, 73.81, 72.55, 73.47, 72.66, 73.28, 70.81, 72.69, 72.46000000000001, 72.59, 73.63, 73.29, 72.67, 73.2, 72.89999999999999, 72.19, 73.9, 73.19]
epsilon=[0.066, 0.1001, 0.2001, 0.3002, 0.4002, 0.5001, 0.6, 0.7001, 0.8004, 0.9003, 1.0003, 1.1024, 1.2, 1.3002, 1.4028, 1.5014, 1.6003, 1.7014, 1.8023, 1.9003, 2.0009, 2.1021, 2.2008, 2.3049, 2.4031, 2.5007, 2.6007, 2.7063, 2.802, 2.901, 3.0006, 3.1065, 3.2063, 3.3032, 3.4005, 3.505, 3.6009, 3.702, 3.8031, 3.9042, 4.0052, 4.101, 4.2054, 4.3003, 4.4051, 4.5001]
sigma=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
clip=[0.1, 1.2139, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
q=[0.001, 0.003, 0.004, 0.002, 0.003, 0.002, 0.002, 0.002, 0.001, 0.002, 0.003, 0.003, 0.001, 0.003, 0.003, 0.001, 0.001, 0.001, 0.003, 0.003, 0.001, 0.001, 0.003, 0.003, 0.001, 0.002, 0.003, 0.003, 0.003, 0.002, 0.003, 0.003, 0.003, 0.002, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.002, 0.002, 0.003, 0.003]
k=[0.5076, 6.6256, 8.8894, 5.6432, 7.5653, 9.6156, 9.103, 9.4447, 9.6583, 9.5729, 9.402, 9.1884, 8.9322, 9.6156, 9.1884, 9.9146, 9.3593, 9.9573, 9.2739, 9.7864, 9.1884, 9.6583, 8.8894, 9.3166, 9.7864, 8.9749, 9.3166, 9.701, 8.8894, 9.2312, 9.5302, 9.8719, 9.0176, 9.3166, 9.5729, 9.8719, 8.9749, 9.2312, 9.4874, 9.7437, 10.0, 9.0603, 9.3166, 9.5302, 9.7437, 9.9573]
theta=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]

# 示例数据
k = np.array(k)  # k数据
theta = np.array(theta)  # theta数据
mia = np.array(mia_lmo)  # mia数据
acc = np.array(acc_lmo)  # acc数据
q = np.array(q)  # q数据
sigma = np.array(sigma)  # sigma数据
epsilon = np.array(epsilon)  # epsilon数据
clip = np.array(clip)  # clip数据

# 准备特征和目标
X1 = np.column_stack((k, theta))  # 用于 f1 的特征
y1 = mia

X2 = np.column_stack((acc, mia, clip, q))  # 用于 f2 的特征
y2 = np.column_stack((k, theta))  # f2 的目标是 k 和 theta

X3 = np.column_stack((epsilon, sigma, clip, q))  # 用于 f3 的特征
y3 = np.column_stack((k, theta))  # f3 的目标是 k 和 theta

# 线性版本实现
# f1: mia = f1(k, theta)
linear_model_f1 = LinearRegression()
linear_model_f1.fit(X1, y1)

# 获取 f1 的公式
a1, b1 = linear_model_f1.coef_
c1 = linear_model_f1.intercept_
print(f"f1 quation: mia = {a1} * k + {b1} * theta + {c1}")

# 定义 f1 预测函数
def f1_linear(k_value, theta_value):
    return a1 * k_value + b1 * theta_value + c1

# f2: (k, theta) = f2(acc, mia, clip, q)
linear_model_f2 = LinearRegression()
linear_model_f2.fit(X2, y2)

# 获取 f2 的公式
a2, b2, c2, d2 = linear_model_f2.coef_[0]
e2, f2, g2, h2 = linear_model_f2.coef_[1]
intercept_k, intercept_theta = linear_model_f2.intercept_
print(f"f2 quation: k = {a2} * acc + {b2} * mia + {c2} * clip + {d2} * q + {intercept_k}")
print(f"theta = {e2} * acc + {f2} * mia + {g2} * clip + {h2} * q + {intercept_theta}")

# 定义 f2 预测函数
def f2_linear(acc_value, mia_value, clip_value, q_value):
    k_pred = a2 * acc_value + b2 * mia_value + c2 * clip_value + d2 * q_value + intercept_k
    theta_pred = e2 * acc_value + f2 * mia_value + g2 * clip_value + h2 * q_value + intercept_theta
    return k_pred, theta_pred

# f3: (k, theta) = f3(epsilon, sigma, clip, q)
linear_model_f3 = LinearRegression()
linear_model_f3.fit(X3, y3)

# 获取 f3 的公式
a3, b3, c3, d3 = linear_model_f3.coef_[0]
e3, f3, g3, h3 = linear_model_f3.coef_[1]
intercept_k_f3, intercept_theta_f3 = linear_model_f3.intercept_
print(f"f3 quation: k = {a3} * epsilon + {b3} * sigma + {c3} * clip + {d3} * q + {intercept_k_f3}")
print(f"theta = {e3} * epsilon + {f3} * sigma + {g3} * clip + {h3} * q + {intercept_theta_f3}")

# 定义 f3 预测函数
def f3_linear(epsilon_value, sigma_value, clip_value, q_value):
    k_pred = a3 * epsilon_value + b3 * sigma_value + c3 * clip_value + d3 * q_value + intercept_k_f3
    theta_pred = e3 * epsilon_value + f3 * sigma_value + g3 * clip_value + h3 * q_value + intercept_theta_f3
    return k_pred, theta_pred

# 非线性版本实现
# f1: mia = f1(k, theta)
rf_model_f1 = RandomForestRegressor(random_state=42)
rf_model_f1.fit(X1, y1)

# 定义 f1 非线性预测函数
def f1_rf(k_value, theta_value):
    return rf_model_f1.predict([[k_value, theta_value]])[0]

# f2: (k, theta) = f2(acc, mia, clip, q)
rf_model_f2 = RandomForestRegressor(random_state=42)
rf_model_f2.fit(X2, y2)

# 定义 f2 非线性预测函数
def f2_rf(acc_value, mia_value, clip_value, q_value):
    pred = rf_model_f2.predict([[acc_value, mia_value, clip_value, q_value]])[0]
    return pred[0], pred[1]

# f3: (k, theta) = f3(epsilon, sigma, clip, q)
rf_model_f3 = RandomForestRegressor(random_state=42)
rf_model_f3.fit(X3, y3)

# 定义 f3 非线性预测函数
def f3_rf(epsilon_value, sigma_value, clip_value, q_value):
    pred = rf_model_f3.predict([[epsilon_value, sigma_value, clip_value, q_value]])[0]
    return pred[0], pred[1]

# 示例使用
print("linear f1:", f1_linear(9.0, 0.0001))
print("non-linear f1:", f1_rf(9.0, 0.0001))
print("linear f2:", f2_linear(72.5, 0.08, 1.0, 0.1))
print("non-linear f2:", f2_rf(72.5, 0.08, 1.0, 0.1))
print("linear f3:", f3_linear(0.1, 0.5, 1.0, 0.1))
print("non-linear f3:", f3_rf(0.1, 0.5, 1.0, 0.1))
