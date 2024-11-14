import sys, time

from util import *


## input params
# load task related params
suffix_dataset=sys.argv[1] # "p100" "fmnist"
suffix_model=sys.argv[2] # "2f" "lr"
suffix_epochs=int(sys.argv[3]) # T
savefolder=sys.argv[4]

## set hyperparams [here distortion is also sigma for gaussian noise]
# parameters in this version
version = "v7"
Qt_filepath="cfg_noise/v4.Qt.mat"
suffix_columns = ["eps", "distortion", "clip", "q", "G_k", "G_theta"]
## load mg noise configs
cfg = load_Qt_mat(Qt_filepath, columns=suffix_columns)
start_time = time.time()
print("now loading configs: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
# print(cfg.shape, '\n', cfg)
cfg = preprocess_df(cfg, remove_nan=True, remove_inf=True)
# print(cfg.shape, '\n', cfg)


for prefix_noise_params in cfg.index:
    epsilon, distortion, clip, q, G_k, G_theta = cfg.loc[int(prefix_noise_params), suffix_columns]
    noise_params = {"a1": 1, "G_k": G_k, "G_theta": G_theta}
    betas, beta_index, beta, tmia, epsilon, delta = theoretical_mia(noise_params, clip, epsilon, alpha=0.2, distributions=["Gamma"], T=suffix_epochs)
    print(tmia)

    filename = f"{version}.mg.{prefix_noise_params}.{suffix_dataset}.{suffix_model}.T_{suffix_epochs}.log"
    filepath = os.path.join(savefolder, filename)
    with open(filepath, 'r', encoding='utf8') as f:
        rs = f.readlines()
    epsilon, distortion, clip, q, G_k, G_theta, accuracy_maintask, accuracy_mia = rs[-4].split("#")[1:]
    cfg.loc[int(prefix_noise_params), ["tmia", "mia", "acc"]] = [tmia, accuracy_mia, accuracy_maintask]

print(cfg)
cfg.to_csv("collect_results.v7.csv")
