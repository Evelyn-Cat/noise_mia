import sys, time

from util import *


## input params
# load task related params
# suffix_dataset=sys.argv[1] # "p100" "fmnist"
# suffix_model=sys.argv[2] # "2f" "lr"
# suffix_epochs=int(sys.argv[3]) # T
# savefolder=sys.argv[4]
# outputfilename=sys.argv[5]
# version=sys.argv[6] # version = "v7"
# machine=sys.argv[7] # 239 becat
suffix_dataset="p100"
suffix_model="2f"
machine = "becat"
savefolder = "logs.becat"


## set hyperparams [here distortion is also sigma for gaussian noise]
# parameters in this version
Qt_filepath="cfg_noise/collect_results.v7.old.csv"
suffix_columns = ["eps", "distortion", "clip", "q", "G_k", "G_theta"]
## load mg noise configs
cfg = load_Qt_mat(Qt_filepath, columns=suffix_columns)
start_time = time.time()
print("now loading configs: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
# print(cfg.shape, '\n', cfg)
cfg = preprocess_df(cfg, remove_nan=True, remove_inf=True)
# print(cfg.shape, '\n', cfg)

for version in ["v6", "v7"]:
    outputfilename = f"collect_results.becat.{version}.T_50_1000.csv"
    for suffix_epochs in [50, 1000]:
        for prefix_noise_params in cfg.index:
            epsilon, distortion, clip, q, G_k, G_theta = cfg.loc[int(prefix_noise_params), suffix_columns]
            noise_params = {"a1": 1, "G_k": G_k, "G_theta": G_theta}
            betas, beta_index, beta, tmia, epsilon, delta = theoretical_mia(noise_params, clip, epsilon, alpha=0.2, distributions=["Gamma"], T=suffix_epochs)
            print(tmia)
            
            filename = f"{version}.mg.{prefix_noise_params}.{suffix_dataset}.{suffix_model}.T_{suffix_epochs}.log"
            filepath = os.path.join(savefolder, filename)
            
            with open(filepath, 'r', encoding='utf8') as f:
                rs = f.readlines()
            try:
                epsilon, distortion, clip, q, G_k, G_theta, accuracy_maintask, accuracy_mia = rs[-4].strip().split("#")[1:]
                cfg.loc[int(prefix_noise_params), ["tmia", "mia", "acc", "T", "version", "machine"]] = [float(tmia), float(accuracy_mia), float(accuracy_maintask), int(suffix_epochs), version, machine]
            except:
                if "exists" in rs[-1]:
                    print(f"please run the inference again for {filename}")
                else:
                    raise NotImplementedError
                
    print(cfg)
    cfg.to_csv(outputfilename)
