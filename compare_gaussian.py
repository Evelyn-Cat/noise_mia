import numpy as np

if __name__ == '__main__':
    # eps=0.1
    # sen=0.1
    # alpha=0.1
    # beta=0.35067041292743084
    # N = {'a1': 0.7, 'a3': 0.2, 'a4': 0.3, 'G_theta': 0.5, 'G_k': 1, 'E_lambda': 5, 'U_b': 2, 'U_a': 1}
    eps=1
    sen=1
    alpha=0.3
    beta=0.2500079170813154
    N = {'a1': 0.8, 'a3': 0.2, 'a4': 0.3, 'G_theta': 1, 'G_k': 2, 'E_lambda': 1, 'U_b': 1, 'U_a': 0, 'mia': 0.49998416583736915}

    # f = np.max([0, 1-delta-np.exp(eps)*alpha, np.exp(-eps) * (1-delta-alpha)]) = beta
    delta1 = 1-beta-np.exp(eps)*alpha
    delta2 = 1-beta/np.exp(-eps)-alpha
    # print(delta1)
    # print(delta2)
    
    # check
    delta = delta1
    f1 = np.max([0, 1-delta-np.exp(eps)*alpha, np.exp(-eps) * (1-delta-alpha)])

    delta = delta2
    f2 = np.max([0, 1-delta-np.exp(eps)*alpha, np.exp(-eps) * (1-delta-alpha)])
    
    # print(f1, f2)
    print(delta2)

    # delta = 0.5388124952650044
    # delta = 0.020408022026964623