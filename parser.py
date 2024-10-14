# eps=0.1
# sen=0.1
# alpha=0.1
# params={'a1': 0.8, 'a4': 0.2, 'G_theta': 0.5, 'G_k': 1, 'U_b': 1, 'U_a': 0, 'mia': 0.0007976317737714567}

# eps=0.1
# sen=0.1
# alpha=0.2
# {'a1': 0.3, 'a4': 0.4, 'G_theta': 0.5, 'G_k': 1, 'U_b': 2, 'U_a': 1, 'mia': 0.009686605458672504}

# eps=0.1
# sen=0.1
# alpha=0.3
# {'a1': 0.5, 'a3': 0.2, 'a4': 0.6, 'G_theta': 3, 'G_k': 2, 'E_lambda': 0.5, 'U_b': 1, 'U_a': 0, 'mia': 0.0005255673419636286}

import os
import matplotlib.pyplot as plt

for filename in os.listdir('results'):
    if filename.startswith("134") and filename.endswith(".log"):
        pass
    elif os.path.exists(".".join([filename.split(".log")[0], "png"])):
        continue
    else:
        continue
    
    splitted = filename.split(".log")[0].split("_")
    eps, sen, alplha = float(splitted[1]), float(splitted[3]), float(splitted[5])

    with open(f"results/{filename}", "r", encoding='utf-8') as f:
        rs = [i.strip() for i in f.readlines()]
    
    results = []
    for r in rs:
        if "a1" in r:
            results.append(eval(r.strip())['mia'])
    
    y = sorted(list(set(results)))
    x = list(range(len(y)))
    
    plt.plot(x, y)
    plt.grid()
    plt.title(f"MIA of {filename}")
    plt.savefig(f'results/{filename.split(".log")[0]}.png')
    plt.cla()
    plt.clf()
