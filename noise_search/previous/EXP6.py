# >>> [a dual grid search] if objective function=> mean(PLRV) fixed => grid search min(MIA)
import numpy as np
import matplotlib.pyplot as plt
from noise.noise_params import distributions, orders
from noise.noise_privacy import compute_rdp_order
from noise.rdp_accounting import get_privacy_spent

# EXP5: vary T
param = {'a1': 0.1, 'a3': 0.1, 'a4': 0.1, 'G_theta': 7.5, 'G_k': 1.0, 'E_lambda': 0.1, 'U_b': 2.0, 'U_a': 1.0, 'epsilon': 2.402, 'mia': 0.0022600000000000398, 'obj': 19.0, 'sensitivity': 1.0}
sigma_Gaussian = 0.49256859462010105

K=0.4
sample_rate=0.05
sensitivity=1.0
Ts=[5, 10, 20, 50, 100]


# delta = compute_delta(param)
delta=1e-10
for T in [1]:
    rdp_Ns = []
    for order in orders:
        rdp_N = compute_rdp_order(param, order, sensitivity)
        rdp_Ns.append(rdp_N * T)
    eps, order = get_privacy_spent(orders=orders, rdp=rdp_Ns, delta=delta)
    print(eps)
    


        




plt.plot(x, noise, label="OurPLRV")
plt.plot(x, noise_Gaussian, label="Gaussian")
plt.title("Fix MIA advantage threshold and Fix sensitivity")
plt.ylabel('Noise Amplitude')
plt.grid()
plt.legend()
plt.savefig("results/noise_visualization.png")

