for alpha in [0.05]:
    for T in [5,10,20,50,100]:
        print(f"python noise_mia.py 1.0 {alpha} {T}")
        print("wait")

for alpha in [0.05]:
    for sen in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(f"python noise_mia.py {sen} {alpha} 1")
        print("wait")
