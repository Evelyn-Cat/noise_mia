"""
This scripts compute the privacy of distribution Laplace(0, b) where the scale parameter b is the noise parameter we sample from distributions.
"""
import numpy as np
from scipy.stats import gamma, expon, uniform
from .noise_params import orders, threshold

# ## When distributions are Gamma, Exponential and Uniform, calculate the RDP.
# def compute_rdp_order(N, order, sensitivity):
#     def compute_M(weight):
#         MGF_1 = ((1-N['a1']*weight*N['G_theta'])**(-N['G_k']))  # Gamma
#         MGF_3 = (N['E_lambda']/(N['E_lambda']-N['a3']*weight))  # Exponential
#         MGF_4 = ((np.exp(N['a4']*weight*N['U_b'])-np.exp(N['a4']*weight*N['U_a']))/(N['a4']*weight*(N['U_b']-N['U_a'])))  # Uniform
#         MGFs = MGF_1 * MGF_3 * MGF_4
#         return MGFs
    
#     try:
#         MGF1 = compute_M(weight=order-1)
#         MGF2 = compute_M(weight=1-sensitivity-order)
#         MGF3 = compute_M(weight=(1-2*sensitivity)*order+(sensitivity-1))
    
#         rdp_N = (1/(order-1)) * np.log((order/(2*order-1)) * MGF1 + (1/2) * MGF2 + (1/(2*(1-2*order))) * MGF3)
#     except:
#         return []
#     return rdp_N


def compute_ma(N, order=1.1, sensitivity=1, distributions=["Gamma", "Exponential", "Uniform"], T=1):
    def compute_M(t, distributions, T=1):
        MGFs = 1
        if "Gamma" in distributions:
            try:
                MGF_Gamma = ((1-N['a1']*t*N['G_theta'])**(-N['G_k']))  # Gamma
            except:
                print(f"MGF_Gamma cannot be computed so we pass this option: {N}")
                return np.nan
            MGFs = MGFs * MGF_Gamma
        elif "Exponential" in distributions:
            try:
                MGF_Exp = (N['E_lambda']/(N['E_lambda']-N['a3']*t))  # Exponential
            except:
                print(f"MGF_Exp cannot be computed so we pass this option: {N}")
                return np.nan
            MGFs = MGFs * MGF_Exp
        elif "Uniform" in distributions:
            try:
                MGF_Uniform = ((np.exp(N['a4']*t*N['U_b'])-np.exp(N['a4']*t*N['U_a']))/(N['a4']*t*(N['U_b']-N['U_a'])))  # Uniform
            except:
                print(f"MGF_Uniform cannot be computed so we pass this option: {N}")
                return np.nan
            MGFs = MGFs * MGF_Uniform
        
        MGFs = MGFs ** T
        return MGFs
    
    try:
        MGF1 = compute_M(t=order*sensitivity, distributions=distributions, T=T)
        MGF2 = compute_M(t=-(order+1)*sensitivity, distributions=distributions, T=T)
        ma_N = np.log(((order+1) * MGF1 + order * MGF2)/(2*order+1))
        print("MGF1, MGF2 or ma_N cannot be computed.")
    except:
        return np.nan
    
    return ma_N


def compute_mia(N, sensitivity=1, epsilon=1, alpha=0.2, distributions=["Gamma", "Exponential", "Uniform"], T=1):
    betas = {}
    for order in orders:
        try:
            ma = compute_ma(N, order=order, sensitivity=sensitivity, distributions=distributions, T=T)
            if ma == np.nan:
                return [], [], [], [], [], []

            delta = np.exp(ma - order * epsilon)
            beta1 = 1 - delta - np.exp(epsilon) * alpha
            beta2 = np.exp(-epsilon) * (1 - delta - alpha)
            
            if not np.isnan(beta1) and not np.isnan(beta2):
                betas[order] = np.max([0, beta1, beta2])
            elif not np.isnan(beta1) and np.isnan(beta2):
                betas[order] = np.max([0, beta1])
            elif np.isnan(beta1) and not np.isnan(beta2):
                betas[order] = np.max([0, beta2])
        except:
            continue
    
    if betas:
        beta_index = max(betas, key=betas.get)
        beta = betas[beta_index]

        mia = 0.5 * (1 - alpha - beta)
        
        return betas, beta_index, beta, mia, epsilon, delta
    else:
        return [], [], [], [], [], []

def compute_obj(N, distributions=["Gamma", "Exponential", "Uniform"], mode=1):
    obj = 0
    if "Gamma" in distributions:
        obj = obj + N["G_k"] * N["G_theta"] if mode==1 else gamma.cdf(threshold["Gamma"], a=N["G_k"], scale=N["G_theta"])
    elif "Exponential" in distributions:
        obj = obj + 1/N['E_lambda'] if mode==1 else expon.cdf(threshold["Exponential"], scale=1/N["E_lambda"])
    elif "Uniform" in distributions:
        obj = obj + (N['U_a'] + N['U_b'])/2 if mode==1 else uniform.cdf(threshold["Uniform"], loc=N["U_a"], scale=N["U_b"]-N["U_a"])
    
    return obj


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
    order = 1.1
    sensitivity = 1
    # rdp_N_ = compute_rdp_order(N, order, sensitivity)
    # print(f"RDP of noise (order={order}, sensitivity={sensitivity}) = {rdp_N_}")


