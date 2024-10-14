"""
This scripts compute the privacy of distribution Laplace(0, b) where the scale parameter b is the noise parameter we sample from distributions.
"""
import numpy as np
from .noise_params import alphas, sensitivity, setted_delta

## When distributions are Gamma and Uniform, calculate the RDP.
def compute_rdp_alpha(N, alpha, sensitivity):
    def compute_M(weight):
        MGF_1 = ((1-N['a1']*weight*N['G_theta'])**(-N['G_k']))  # Gamma
        MGF_4 = ((np.exp(N['a4']*weight*N['U_b'])-np.exp(N['a4']*weight*N['U_a']))/(N['a4']*weight*(N['U_b']-N['U_a'])))  # Uniform
        MGFs = MGF_1 * MGF_4
        return MGFs
    
    MGF1 = compute_M(weight=alpha-1)
    MGF2 = compute_M(weight=1-sensitivity-alpha)
    MGF3 = compute_M(weight=(1-2*sensitivity)*alpha+(sensitivity-1))
    
    rdp_N = (1/(alpha-1)) * np.log((alpha/(2*alpha-1)) * MGF1 + (1/2) * MGF2 + (1/(2*(1-2*alpha))) * MGF3)
    return rdp_N


def compute_ma(N, alpha, sensitivity):
    def compute_M(weight):
        MGF_1 = ((1-N['a1']*weight*N['G_theta'])**(-N['G_k']))  # Gamma
        MGF_4 = ((np.exp(N['a4']*weight*N['U_b'])-np.exp(N['a4']*weight*N['U_a']))/(N['a4']*weight*(N['U_b']-N['U_a'])))  # Uniform
        MGFs = MGF_1 * MGF_4
        return MGFs
    
    MGF1 = compute_M(weight=alpha*sensitivity)
    MGF2 = compute_M(weight=-(alpha+1)*sensitivity)
    
    ma_N = np.log(((alpha+1)/(2*alpha+1)) * MGF1 + (alpha/(2*alpha+1)) * MGF2)
    return ma_N


def compute_mia(N, sensitivity, epsilon):
    betas = {}
    for alpha in alphas:
        try:
            beta1 = 1 - np.exp(compute_ma(N, alpha, sensitivity) - alpha * epsilon) - np.exp(epsilon) * alpha
            beta2 = np.exp(-epsilon) * (1 - np.exp(compute_ma(N, alpha, sensitivity) - alpha * epsilon) - alpha)
            if not np.isnan(beta1) and not np.isnan(beta2):
                betas[alpha] = np.max(0, beta1, beta2)
        except:
            continue
    
    if betas:
        beta_index = np.max(betas, key=betas.get)
        beta = betas[beta_index]
        
        mia = 1 - 2 * beta
        return betas, beta_index, beta, mia
    else:
        return [], [], [], []


if __name__ == '__main__':
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
    alpha = 0.3
    sensitivity = 1
    rdp_N_ = compute_rdp_alpha(N, alpha, sensitivity)
    print(f"RDP of noise (alpha={alpha}, sensitivity={sensitivity}) = {rdp_N_}")


