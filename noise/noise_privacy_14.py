"""
This scripts compute the privacy of distribution Laplace(0, b) where the scale parameter b is the noise parameter we sample from distributions.
"""
import numpy as np
from .noise_params import orders, sensitivity

## When distributions are Gamma, Exponential and Uniform, calculate the RDP.
def compute_rdp_order(N, order, sensitivity):
    def compute_M(weight):
        MGF_1 = ((1-N['a1']*weight*N['G_theta'])**(-N['G_k']))  # Gamma
        MGF_4 = ((np.exp(N['a4']*weight*N['U_b'])-np.exp(N['a4']*weight*N['U_a']))/(N['a4']*weight*(N['U_b']-N['U_a'])))  # Uniform
        MGFs = MGF_1 * MGF_4
        return MGFs
    
    MGF1 = compute_M(weight=order-1)
    MGF2 = compute_M(weight=1-sensitivity-order)
    MGF3 = compute_M(weight=(1-2*sensitivity)*order+(sensitivity-1))
    
    rdp_N = (1/(order-1)) * np.log((order/(2*order-1)) * MGF1 + (1/2) * MGF2 + (1/(2*(1-2*order))) * MGF3)
    return rdp_N


def compute_ma(N, order, sensitivity):
    def compute_M(weight):
        MGF_1 = ((1-N['a1']*weight*N['G_theta'])**(-N['G_k']))  # Gamma
        MGF_4 = ((np.exp(N['a4']*weight*N['U_b'])-np.exp(N['a4']*weight*N['U_a']))/(N['a4']*weight*(N['U_b']-N['U_a'])))  # Uniform
        # print(f"MGF_1:{MGF_1}")
        # print(f"MGF_4:{MGF_4}")
        MGFs = MGF_1 * MGF_4
        return MGFs
    
    MGF1 = compute_M(weight=order*sensitivity)
    MGF2 = compute_M(weight=-(order+1)*sensitivity)
    # print(f"mgf1:{MGF1} order:{order} sen:{sensitivity}")
    # print(f"mgf2:{MGF2} order:{order} sen:{sensitivity}")
    
    i01 = ((order+1)/(2*order+1)) * MGF1
    i02 = (order/(2*order+1)) * MGF2
    # print(f"i01:{i01}")
    # print(f"i02:{i02}")
    i1 = ((order+1)/(2*order+1)) * MGF1 + (order/(2*order+1)) * MGF2
    # print(f"i1:{i1}")
    ma_N = np.log(((order+1)/(2*order+1)) * MGF1 + (order/(2*order+1)) * MGF2)
    # print(f"ma_N:{ma_N}")
    return ma_N


def compute_mia(N, sensitivity, epsilon, alpha):
    betas = {}
    for order in orders:
        try:
            # print(f"order: {order} sensitivity: {sensitivity}")
            ma = compute_ma(N, order, sensitivity)
            # print(f"ma:{ma}")
            delta = np.exp(compute_ma(N, order, sensitivity) - order * epsilon)
            # print(f"delta:{delta}")
            beta1 = 1 - delta - np.exp(epsilon) * alpha
            beta2 = np.exp(-epsilon) * (1 - delta - alpha)
            # print(f"beta1:{beta1}")
            # print(f"beta2:{beta2}")
            # print("\n")
            if not np.isnan(beta1) and not np.isnan(beta2):
                betas[order] = np.max([0, beta1, beta2])
        except:
            continue
    
    # print(f"betas:{betas}")
    if betas:
        beta_index = max(betas, key=betas.get)
        beta = betas[beta_index]
        # print(f"beta:{beta}")
        mia = 1 - 2 * beta
        # print(f"mia:{mia}")
        return betas, beta_index, beta, mia
    else:
        return [], [], [], []


if __name__ == '__main__':
    N = {
        "G_k": 1.5,
        "G_theta": 0.1,
        "U_a": 0,
        "U_b": 1,
        "a1": 1,
        "a4": 0.1
    }
    order = 0.3
    sensitivity = 1
    rdp_N_ = compute_rdp_order(N, order, sensitivity)
    print(f"RDP of noise (order={order}, sensitivity={sensitivity}) = {rdp_N_}")


