import numpy as np


## sample noises
def generate_noise(N, distributions, noise_size):
    sampled_parameters = sample_parameter(N, distributions, noise_size)
    return np.random.laplace(0, sampled_parameters)


## sample parameters from the given noise parameters and distributions
def sample_parameter(N, distributions, noise_size=1):
    us = 0
    
    if "Gamma" in distributions:
        us = us + N['a1']*np.random.gamma(N["G_k"], N["G_theta"], noise_size)
    else:
        us = us + 0
    
    if "Exponential" in distributions:
        us = us + N['a3']*np.random.exponential(N["E_lambda"], noise_size)
    else:
        us = us + 0
    
    if "Uniform" in distributions:
        us = us + N['a4'] * np.random.uniform(N["U_a"], N["U_b"], noise_size)
    else:
        us = us + 0
    
    return 1/us


if __name__ == "__main__":
    N = {
        "G_k": 1.5,
        "G_theta": 0.1,
        "E_lambda": 0.5,
        "U_a": 0,
        "U_b": 1,
        "a1": 1,
        "a3": 0.5,
        "a4": 0.1
    }
    distributions = ["Gamma", "Exponential", "Uniform"]
    noise_size = 10
    noisy_data = generate_noise(N, distributions, noise_size)
    
    print(f"Generated {noise_size} noisy data samples.")
    print(noisy_data)
