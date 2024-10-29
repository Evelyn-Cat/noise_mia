import numpy as np
from scipy.stats import norm
from noise_mia.noise.noise_params_134 import orders

# f-DP for Gaussian: 0.5 - alpha - u-GDP(sigma) < K
def compute_obj_Gaussian(K, alpha, sensitivity):
    G_lb = 0.5 - alpha - K  # 0.5 - alpha - G_lb [=beta(sigma)] < K
    mu = norm.ppf(1-alpha) - norm.ppf(G_lb)
    sigma = sensitivity / mu
    obj = mu * mu / 2
    return obj, sigma
