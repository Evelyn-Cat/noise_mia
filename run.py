for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for sen in [0.1, 0.5, 1]:
        for eps in [0.1, 0.5, 1, 5, 10]:
            print(f"python noise_mia_14.py {eps} {sen} {alpha} > results/14.eps_{eps}_sen_{sen}_alpha_{alpha}.log")
            print("wait")
