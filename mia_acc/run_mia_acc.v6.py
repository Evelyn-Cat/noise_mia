import sys, copy, time
from scipy.special import softmax
from sklearn.metrics import accuracy_score

from util import *


## input params
# load task related params
suffix_dataset=sys.argv[1] # "p100" "fmnist"
suffix_model=sys.argv[2] # "2f" "lr"
suffix_epochs=int(sys.argv[3])
savefolder=sys.argv[4]
# load noise related params
prefix_noise_type=sys.argv[5] # "mg" "gaussian"
prefix_noise_params=sys.argv[6]
# different machine
data_dir="../noise_auditing/auditing/datasets" # sys.argv[9]

savefolder = os.path.join(savefolder, suffix_dataset, suffix_model)
if not os.path.exists(savefolder):
    os.makedirs(savefolder)

## set hyperparams [here distortion is also sigma for gaussian noise]
# parameters in this version
version = "v6"
Qt_filepath="cfg_noise/v4.Qt.mat"
suffix_columns = ["eps", "distortion", "clip", "q", "G_k", "G_theta"]
## load mg noise configs
cfg = load_Qt_mat(Qt_filepath, columns=suffix_columns)
start_time = time.time()
print("now loading configs: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
# print(cfg.shape, '\n', cfg)
cfg = preprocess_df(cfg, remove_nan=True, remove_inf=True)
# print(cfg.shape, '\n', cfg)
epsilon, distortion, clip, q, G_k, G_theta = cfg.loc[int(prefix_noise_params), suffix_columns]

if suffix_dataset == "p100":
    microbatches=1
    batchsize=int(10000*q)
elif suffix_dataset == "fmnist":
    microbatches=int(6000*q)
    batchsize=microbatches
    exit(0)
else:
    raise NotImplementedError


# load dataset and exp name
tasktype, data, learning_rate = read_dataset(suffix_dataset, data_dir)
trn_x, trn_y = data['trn']
tst_x, tst_y = data['tst']

exp_suffix = ".h5" if tasktype == "cv" else ".safetensors"
exp_name = f"{version}.{prefix_noise_type}.{prefix_noise_params}.{suffix_dataset}.{suffix_model}.T_{suffix_epochs}"
savepath = os.path.join(savefolder, exp_name+exp_suffix)
if os.path.exists(savepath):
    print(f"{savepath} exists.")
    exit(0)

if prefix_noise_type == "mg":
    prefix_noise_params = [prefix_noise_type, {"a1": 1, "G_k": G_k, "G_theta": G_theta}]
else:
    prefix_noise_params = float(prefix_noise_params)


# train model and save models
print(prefix_noise_params)
mia_acc = mia_acc_CV(suffix_dataset,suffix_model,suffix_epochs,microbatches,learning_rate,batchsize,clip,prefix_noise_type,prefix_noise_params)
model = mia_acc.build_model(trn_x, trn_y)

np.random.seed(None)
new_seed = np.random.randint(1000000)
mia_acc.train_model(model, trn_x, trn_y, trn_x, trn_y, savepath, new_seed)


# run main task and return main task accuracy
accuracy_all = copy.deepcopy(cfg)

model = tf.keras.models.load_model(savepath)
tst_preds = softmax(model.predict(tst_x), axis=1)
predict_single_label = np.argmax(tst_preds, axis=1)
true_single_label = np.argmax(tst_y, axis=1)
accuracy_maintask = accuracy_score(true_single_label, predict_single_label)

# run mia task and return mia task accuracy
accuracy_mia = mia(model, trn_x, trn_y, tst_x, tst_y)

# output main and mia task accuracy
if prefix_noise_type == "mg":
    out_line=f"print_results#{epsilon}#{distortion}#{clip}#{q}#{G_k}#{G_theta}#{accuracy_maintask}#{accuracy_mia}"
elif prefix_noise_type == "gaussian":
    out_line=f"print_results#{prefix_noise_params}#{accuracy_maintask}#{accuracy_mia}"
print(out_line)


end_time = time.time()
elapsed_time = end_time - start_time
print("Program starts:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
print("Program ends:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
print(f"Program runs {elapsed_time} seconds.")

