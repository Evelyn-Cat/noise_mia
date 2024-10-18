import numpy as np
import matplotlib.pyplot as plt
from noise.noise_privacy import compute_obj_Gaussian
from noise.noise_params import distributions, orders
from noise.noise_generation import compute_rdp_order
from noise.rdp_accounting import get_privacy_spent

# EXP5: vary T
param = {'a1': 0.1, 'a3': 0.1, 'a4': 0.1, 'G_theta': 7.5, 'G_k': 1.0, 'E_lambda': 0.1, 'U_b': 2.0, 'U_a': 1.0, 'epsilon': 2.402, 'mia': 0.0022600000000000398, 'obj': 19.0, 'sensitivity': 1.0, 'delta': 0.2}
sigma_Gaussian = 0.49256859462010105

K=0.4
alpha=0.05
sample_rate=0.05
sensitivity=1.0
Ts=[5, 10, 20, 50, 100]

acc = [[], []]
for T in Ts:
    obj = param['obj']
    
    mu = sensitivity / sigma_Gaussian
    obj_Gaussian = mu * mu / 2

    acc[0].append(obj)
    acc[1].append(obj_Gaussian)


plt.plot(Ts, acc[0], label="OurPLRV")
plt.plot(Ts, acc[1], label="Gaussian")
plt.title("Fix MIA advantage threshold Fix sensitivity and Vary T")
plt.xlabel('MIA threshold')
plt.ylabel('Accuracy (mean of epsilon)')
plt.grid()
plt.legend()
plt.savefig("results/noise_visualization.png")

