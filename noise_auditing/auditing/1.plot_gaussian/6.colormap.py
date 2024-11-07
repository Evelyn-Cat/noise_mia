from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
np.random.seed(None)
from sklearn.metrics import accuracy_score
from scipy.special import softmax



import tensorflow as tf
from distutils.version import LooseVersion
if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
    GradientDescentOptimizer = tf.train.GradientDescentOptimizer
    AdamOptimizer = tf.train.AdamOptimizer
else:
    GradientDescentOptimizer = tf.optimizers.SGD
    AdamOptimizer = tf.optimizers.Adam





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

def data(data_dir, dataset):
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
    
    return train_x, train_y, tst_x, tst_y 



data_dir = "../../datasets"
dataset = "p100"
trn_x, trn_y, tst_x, tst_y = data(data_dir)



filename = "/Users/mac/Documents/PRO/noise_mia/noise_auditing/auditing/Qt.xlsx"
df = pd.read_excel(filename)
dataset = "p100"
model = "2f"




noise_type = "lmo"
modelfolder = "/Users/mac/Documents/PRO/noise_mia/save_results/auditing/cv/p100_2f"

# noise_type = "gaussian"
# modelfolder = "/Users/mac/Documents/PRO/noise_mia/save_results.check/auditing/cv/p100_2f"




Ts = [1, 5, 10, 50, 100, 1000]
for T in Ts:
    for config_idx in range(0, 90):
        epsilon, sigma, clip, q, k, theta = df.loc[config_idx, :]
        print(epsilon)
        
        modelname = f"{dataset}-{model}-{noise_type}-{config_idx}-{T}-{q}.h5"
        modelpath = os.path.join(modelfolder, modelname)
        
        if not os.path.exists(modelpath):
            print(f"Model {modelname} not exists.")
            continue
        else:
            print(f"{T}, {config_idx}:")
            acc = mia(modelpath, trn_x, trn_y, tst_x, tst_y)
            print(f"print_results: {acc}")
            
