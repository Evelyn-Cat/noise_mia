import os, sys
import numpy as np
np.random.seed(None)
from absl import app
from scipy.special import softmax

import tensorflow as tf
from distutils.version import LooseVersion
if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
    GradientDescentOptimizer = tf.train.GradientDescentOptimizer
    AdamOptimizer = tf.train.AdamOptimizer
else:
    GradientDescentOptimizer = tf.optimizers.SGD
    AdamOptimizer = tf.optimizers.Adam

def main(_):    
    dataset="p100"
    model="2f"
    noise_type="gaussian"
    noise_params = sys.argv[1]
    epochs=100
    l2_norm_clip=1.0
    modelname = f"{dataset}-{model}-{noise_type}-{noise_params}-{epochs}-{l2_norm_clip}.h5"
    modelfolder = "/root/new_lmo/noise_mia/save_results/auditing/cv/p100_2f"
    modelpath = os.path.join(modelfolder, modelname)

    ## read dataset
    data_dir = "../datasets"
    if "fmnist" in dataset:
        loader = np.load(os.path.join(data_dir, "fmnist/clipbkd-new-1.npy"), allow_pickle=True)
        all_bkds = {"tst": loader[3],"trn": loader[0]}
        all_bkds["tst"] = all_bkds["tst"][0].reshape((-1, 28, 28, 1)), np.eye(2)[all_bkds["tst"][1]]
        all_bkds["trn"] = all_bkds["trn"][0].reshape((-1, 28, 28, 1)), np.eye(2)[all_bkds["trn"][1]]
    elif "p100" in dataset:
        loader = np.load(os.path.join(data_dir, "p100/p100_1.npy"), allow_pickle=True)
        all_bkds = {"tst": loader[3],"trn": loader[0]}
        all_bkds["tst"] = all_bkds["tst"][0].reshape((-1, 100)), np.eye(100)[all_bkds["tst"][1]]
        all_bkds["trn"] = all_bkds["trn"][0].reshape((-1, 100)), np.eye(100)[all_bkds["trn"][1]]

    train_x, train_y = all_bkds['trn']
    tst_x, tst_y = all_bkds['tst']
    
    np.random.seed(42)
    sample_fraction = 0.1
    sample_indices = np.random.choice(len(train_x), int(len(train_x) * sample_fraction), replace=False)
    trn_x = train_x[sample_indices]
    trn_y = train_y[sample_indices]
    # print(train_y.shape)
    # print(trn_y.shape)
    # print(tst_y.shape)
    
    
    ## compute mia acc and save
    def mia(modelpath, trn_x, trn_y, tst_x, tst_y):
        np.random.seed(42)
        length = np.min([tst_y.shape[0], trn_y.shape[0]])
        tst_y_inds = np.random.choice(tst_y.shape[0], length, replace=False)
        tst_x, tst_y = tst_x[tst_y_inds], tst_y[tst_y_inds]
        
        model = tf.keras.models.load_model(modelpath)

        trn_preds = softmax(model.predict(trn_x), axis=1)
        tst_preds = softmax(model.predict(tst_x), axis=1)
        trn_loss = np.multiply(trn_preds, trn_y).sum(axis=1)
        tst_loss = np.multiply(tst_preds, tst_y).sum(axis=1)
        
        trn_loss_mean = trn_loss.mean()
        trn_thresh = (trn_preds >= trn_loss_mean).sum()
        tst_thresh = length - (tst_preds >= trn_loss_mean).sum()
        acc = (trn_thresh + tst_thresh) / length
        return np.log(acc)
    
    nob_vals = mia(modelpath, trn_x, trn_y, tst_x, tst_y)
    print(modelname, nob_vals)
    

    
if __name__ == '__main__':
    app.run(main)
    