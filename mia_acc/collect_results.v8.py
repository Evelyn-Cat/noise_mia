import sys, time

from util import *
###### noise_type: mg noise

## input params
suffix_dataset="p100" # sys.argv[1] # "p100" "fmnist"
suffix_model="2f" # sys.argv[2] # "2f" "lr"
version="v8"
G_k=2.05980287646780
G_theta=1e-5
clip=0.1
savefolder="logs"

machine="becat"

Ts = [10, 20, 30, 40, 50, 100, 200, 500, 1000]
# Ts = [1000]
suffix_epochs = 10
# outputfilename="collect_results.becat.v8.csv"

cfg = pd.DataFrame([])
start_time = time.time()

cnt = 0
for suffix_epochs in Ts:
    outputfilename=f"collect_results.becat.v8.T_{suffix_epochs}.acc+1.csv"
    for rp in range(10):
        noise_type = "mg"
        filename = f"rp_{rp}.{version}.mg.0.{suffix_dataset}.{suffix_model}.T_{suffix_epochs}.log"
        filepath = os.path.join(savefolder, filename)
        
        with open(filepath, 'r', encoding='utf8') as f:
            rs = f.readlines()
        try:
            # epsilon, distortion, clip, q, G_k, G_theta, accuracy_maintask, accuracy_mia = rs[-4].strip().split("#")[1:]
            _, q, _, _, accuracy_maintask, accuracy_mia = rs[-4].strip().split("#")[1:]
            print(clip, q, G_k, G_theta, accuracy_maintask, accuracy_mia)
            cfg.loc[cnt, ["noise_type", "mia", "acc", "T", "version", "machine", "rp"]] = [0, float(accuracy_mia), float(accuracy_maintask), int(suffix_epochs), version, machine, rp]
            cnt = cnt + 1
        except:
            if "exists" in rs[-1]:
                print(f"please run the inference again for {filename}")
            else:
                raise NotImplementedError
        
        noise_type = "gaussian"
        for sigma in [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 100, 1000, 10000]:
            filename = f"rp_{rp}.{version}.gaussian.{sigma}.{suffix_dataset}.{suffix_model}.T_{suffix_epochs}.log"
            filepath = os.path.join(savefolder, filename)
            with open(filepath, 'r', encoding='utf8') as f:
                rs = f.readlines()

            try:
                # epsilon, distortion, clip, q, G_k, G_theta, accuracy_maintask, accuracy_mia = rs[-4].strip().split("#")[1:]
                _, accuracy_maintask, accuracy_mia = rs[-4].strip().split("#")[1:]
                print(sigma, accuracy_maintask, accuracy_mia)
                cfg.loc[cnt, ["noise_type", "mia", "acc", "T", "version", "machine", "rp"]] = [sigma, float(accuracy_mia), float(accuracy_maintask), int(suffix_epochs), version, machine, rp]
                cnt = cnt + 1
            except:
                pass
                # if "exists" in rs[-1]:
                #     print(f"please run the inference again for {filename}")
                # else:
                #     raise NotImplementedError
        
    print(cfg)
    cfg.to_csv(outputfilename)
