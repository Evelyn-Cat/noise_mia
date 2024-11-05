from noise_mia.noise.noise_params_134 import noise_params, distributions
from noise.noise_generation import generate_noise

noise_size = 10
noisy_data = generate_noise(noise_params[0.3], distributions, noise_size)

