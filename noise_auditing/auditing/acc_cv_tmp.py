# import sys
import numpy as np
from sklearn.metrics import accuracy_score

import os, sys
import numpy as np
np.random.seed(None)
from scipy.special import softmax
from collections import defaultdict

import tensorflow as tf
from distutils.version import LooseVersion
if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
    GradientDescentOptimizer = tf.train.GradientDescentOptimizer
    AdamOptimizer = tf.train.AdamOptimizer
else:
    GradientDescentOptimizer = tf.optimizers.SGD
    AdamOptimizer = tf.optimizers.Adam

from audit_cv import auditing_CV


if __name__ == '__main__':
    noise_type = "gaussian" if len(sys.argv) > 1 else "mia_guard"
    # dataset = sys.argv[2] if len(sys.argv) > 2 else "fmnist"
    # model = sys.argv[3] if len(sys.argv) > 3 else "lr"
    # run_if = True if len(sys.argv) > 4 else False
    
    num = 10
    clip_norms = 1 # sensitivity
    
    noise_params = {}
    if noise_type == "mia_guard":
        filepath = f"/mnt/nvme1n1p1/home/qiy22005/PRO/noise_mia/save_results.2024.10.22/auditing/cv/fmnist_2f/fmnist_2f-nbkd-lmo-2.402-1-{num}.h5"
    elif noise_type == "gaussian":
        filepath = f"/mnt/nvme1n1p1/home/qiy22005/PRO/noise_mia/save_results.2024.10.22/auditing/cv/fmnist_2f/fmnist_2f-nbkd-gaussian-0.49256859462010105-1-{num}.h5"
    
    
    dataset = "fmnist"
    data_dir = "/mnt/nvme1n1p1/home/qiy22005/PRO/noise_mia/noise_auditing/auditing/datasets"
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
    
    
    trn_x, trn_y = all_bkds['trn']
    tst_x, tst_y = all_bkds['tst']
    
    auditor = auditing_CV()
    auditor.load_data()
    auditor.build_model()
    auditor.train_model()
    
    print(noise_type)
    model = tf.keras.models.load_model(filepath)
    tst_preds = softmax(model.predict(tst_x), axis=1)
    print(tst_preds.shape)
    print(tst_y.shape)
    
    predict_single_label = np.argmax(tst_preds, axis=1)
    true_single_label = np.argmax(tst_y, axis=1)
    accuracy = accuracy_score(true_single_label, predict_single_label)
    print(accuracy)