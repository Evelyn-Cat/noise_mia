import numpy as np
import matplotlib.pyplot as plt
from noise.noise_params import distributions

# EXP4: the optimal noise visualization.
from noise.noise_generation import generate_noise
noise_size=1000
noise_generations = [[], []]
# param = {'a1': 0.1, 'a3': 0.1, 'a4': 0.1, 'G_theta': 7.5, 'G_k': 1.0, 'E_lambda': 0.1, 'U_b': 2.0, 'U_a': 1.0, 'epsilon': 2.402, 'mia': 0.0022600000000000398, 'obj': 19.0, 'sensitivity': 1.0}
# sigma_Gaussian = 0.49256859462010105
# param = {'a1': 0.1, 'a4': 0.1, 'G_theta': 3.0, 'G_k': 2.0, 'U_b': 2.0, 'U_a': 1.0, 'epsilon': 2.402, 'mia': 0.00226, 'obj': 7.5, 'delta': 0.0, 'sensitivity': 1.0}
# sigma_Gaussian = 0.30397841595588454
param = []
sigma_Gaussian = 0.1

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
plt.savefig("results/noise_visualization.png")

