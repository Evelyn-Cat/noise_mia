import os
import matplotlib.pyplot as plt

def parser_file(filename):
    with open(filename, "r", encoding='utf-8') as f:
        rs = [i.strip().split("\t") for i in f.readlines()]
    content = [[float(f"{float(j):.5f}") for j in i] for i in rs[1:]]
    return rs[0], content

if __name__ == '__main__':
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
