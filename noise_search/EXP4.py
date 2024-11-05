import numpy as np
import matplotlib.pyplot as plt

dists = "geu"
alpha = 0.15

distributions = []
if "g" in dists:
    distributions.append("Gamma")
if "e" in dists:
    distributions.append("Exponential")
if "u" in dists:
    distributions.append("Uniform")


# EXP4: the optimal noise visualization.
from noise.noise_generation import generate_noise
noise_size=1000
noise_generations = [[], []]
# param = {'a1': 0.1, 'a3': 0.1, 'a4': 0.1, 'G_theta': 7.5, 'G_k': 1.0, 'E_lambda': 0.1, 'U_b': 2.0, 'U_a': 1.0, 'epsilon': 2.402, 'mia': 0.0022600000000000398, 'obj': 19.0, 'sensitivity': 1.0}
# sigma_Gaussian = 0.49256859462010105
# param = {'a1': 0.1, 'a4': 0.1, 'G_theta': 3.0, 'G_k': 2.0, 'U_b': 2.0, 'U_a': 1.0, 'epsilon': 2.402, 'mia': 0.00226, 'obj': 7.5, 'delta': 0.0, 'sensitivity': 1.0}
# sigma_Gaussian = 0.30397841595588454

# [exp1] alpha: 0.2; sensitivity: 1;
# param = {'a1': 0.1, 'a3': 0.1, 'a4': 0.1, 'G_theta': 7.5, 'G_k': 1.0, 'E_lambda': 0.1, 'U_b': 2.0, 'U_a': 1.0, 'epsilon': 0.10967, 'mia': 0.01351, 'obj': 19.0, 'delta': 0.3416, 'sensitivity': 1.0}
# sigma_Gaussian = 0.4709932231684791
# # [exp1] alpha: 0.15; sensitivity: 1;
# param = {'a1': 0.1, 'a3': 0.1, 'a4': 0.1, 'G_theta': 7.5, 'G_k': 1.0, 'E_lambda': 0.1, 'U_b': 2.0, 'U_a': 1.0, 'epsilon': 0.10967, 'mia': 0.01061, 'obj': 19.0, 'delta': 0.3416, 'sensitivity': 1.0}
# sigma_Gaussian = 0.48242367051124024

if dists == "g":
    ## 2024.10.31 alpha: 0.2; sensitivity: 1; K: 0.005; from EXP1
    param = {'a1': 1.0, 'G_theta': 10.0, 'G_k': 10.0, 'epsilon': 0.01, 'mia': 0.001, 'delta': 0.0, 'obj1': 100.0, 'obj2': 0.0, 'sensitivity': 1.0}
    sigma_Gaussian = 0.7243976515506757
    ## 2024.10.31 alpha: 0.15; sensitivity: 1; K: 0.005; from EXP1
    param = {'a1': 1.0, 'G_theta': 10.0, 'G_k': 10.0, 'epsilon': 0.01, 'mia': 0.001, 'delta': 0.0, 'obj1': 100.0, 'obj2': 0.0, 'sensitivity': 1.0}
    sigma_Gaussian = 0.6967240601856282

if dists == "geu":
    ## 2024.10.31 alpha: 0.2; sensitivity: 1; K: 0.005; from EXP1
    param = {'a1': 0.1, 'a3': 0.1, 'a4': 0.1, 'G_theta': 3.0, 'G_k': 2.0, 'E_lambda': 0.1, 'U_b': 1.0, 'U_a': 0.0, 'epsilon': 0.01, 'mia': 0.001, 'delta': 0.0, 'obj1': 6.0, 'obj2': 0.001, 'sensitivity': 1.0}
    sigma_Gaussian = 0.7243976515506757
    ## 2024.10.31 alpha: 0.15; sensitivity: 1; K: 0.005; from EXP1
    param = {'a1': 0.1, 'a3': 0.1, 'a4': 0.1, 'G_theta': 3.0, 'G_k': 2.0, 'E_lambda': 0.1, 'U_b': 1.0, 'U_a': 0.0, 'epsilon': 0.01, 'mia': 0.001, 'delta': 0.0, 'obj1': 6.0, 'obj2': 0.001, 'sensitivity': 1.0}
    sigma_Gaussian = 0.6967240601856282


noise = generate_noise(param, distributions, noise_size=noise_size)
# noise = generate_noise(param, ["Gamma", "Uniform"], noise_size=noise_size)
noise_Gaussian = np.random.normal(sigma_Gaussian, size=noise_size)
x = list(range(len(noise)))

plt.plot(x, noise, label="OurPLRV")
plt.plot(x, noise_Gaussian, label="Gaussian")
plt.title("Fix MIA advantage threshold and Fix sensitivity")
plt.ylabel('Noise Amplitude')
plt.grid()
plt.legend()
plt.savefig(f"results/{dists}.exp4.alpha={alpha}.noise_visualization.png")

