from matplotlib import pyplot as plt

results = {
    ("fmnist", "2f", "gaussian", 0.6967240601856282, 1):    0.961,
    ("fmnist", "2f", "gaussian", 0.6967240601856282, 5):    0.9805,
    ("fmnist", "2f", "gaussian", 0.6967240601856282, 10):   0.9845,
    ("fmnist", "2f", "gaussian", 0.6967240601856282, 20):   0.9855,
    ("fmnist", "2f", "gaussian", 0.6967240601856282, 50):   0.9855,
    ("fmnist", "2f", "gaussian", 0.6967240601856282, 100):  0.9895,
    ("fmnist", "2f", "gaussian", 0.7243976515506757, 1):    0.9565,
    ("fmnist", "2f", "gaussian", 0.7243976515506757, 5):    0.9815,
    ("fmnist", "2f", "gaussian", 0.7243976515506757, 10):   0.984,
    ("fmnist", "2f", "gaussian", 0.7243976515506757, 20):   0.986,
    ("fmnist", "2f", "gaussian", 0.7243976515506757, 50):   0.988,
    ("fmnist", "2f", "gaussian", 0.7243976515506757, 100):  0.989,

    ("fmnist", "2f", "mia_guard", 0.6967240601856282, 1):    0.961,
    ("fmnist", "2f", "mia_guard", 0.6967240601856282, 5):    0.9805,
    ("fmnist", "2f", "mia_guard", 0.6967240601856282, 10):   0.9845,
    ("fmnist", "2f", "mia_guard", 0.6967240601856282, 20):   0.9855,
    ("fmnist", "2f", "mia_guard", 0.6967240601856282, 50):   0.9855,
    ("fmnist", "2f", "mia_guard", 0.6967240601856282, 100):  0.9895,
    ("fmnist", "2f", "mia_guard", 0.7243976515506757, 1):    0.9565,
    ("fmnist", "2f", "mia_guard", 0.7243976515506757, 5):    0.9815,
    ("fmnist", "2f", "mia_guard", 0.7243976515506757, 10):   0.984,
    ("fmnist", "2f", "mia_guard", 0.7243976515506757, 20):   0.986,
    ("fmnist", "2f", "mia_guard", 0.7243976515506757, 50):   0.988,
    ("fmnist", "2f", "mia_guard", 0.7243976515506757, 100):  0.989,
}


dataset = "fmnist"
model = "2f"
noise_type = "gaussian"
sigma1 = 0.6967240601856282
sigma2 = 0.7243976515506757
Ts = [1, 5, 10, 20, 50, 100]

color = ['r--', 'bs', 'g^']
acc = {sigma1:[], sigma2:[]}
for cnt, sigma in enumerate(acc):
    for T in Ts:
        idx = (dataset, model, noise_type, sigma, T)
        acc[sigma].append(results[idx])
    
    plt.plot(Ts, acc[sigma], color[cnt], label="sigma={:.4f}".format(sigma))
    plt.xlabel("T")
    plt.ylabel("Accuracy")
plt.grid()
plt.legend(loc="lower right")
plt.savefig(f"{dataset}.{model}.{noise_type}.png")

