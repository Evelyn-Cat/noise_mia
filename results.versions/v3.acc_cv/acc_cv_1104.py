from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
np.random.seed(None)
from sklearn.metrics import accuracy_score
from scipy.special import softmax


from absl import app
from absl import flags



flags.DEFINE_string('dataset', 'fmnist', 'fmnist, p100, sst2, qnli.')
flags.DEFINE_string('model', '2f', '[fmnist, p100:] 2f, lr; [sst2, qnli:] r, b.')
flags.DEFINE_integer('n_pois', 8, '[number of clusters:] 1, 2, 4, 8.')
flags.DEFINE_float('l2_norm_clip', 1.0, '[Clipping norm] 1')
flags.DEFINE_string('exp_name', None, '[name of experiment] dataset, model, n/bkd, clip_norm, noise_type, noise_param, trial')
flags.DEFINE_string('noise_type', 'gaussian', '[type of noise] gaussian, lmo')
flags.DEFINE_float('noise_params', 1.1, '[For gaussian: ratio of the standard deviation to the clipping norm; For lmo: lmo params index]')
flags.DEFINE_boolean('backdoor', False, '[whether to backdoor] False, True.')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_integer('microbatches', 1, 'Number of microbatches (must evenly divide batch_size)')
flags.DEFINE_integer('epochs', 24, 'Number of epochs')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
FLAGS = flags.FLAGS



import tensorflow as tf
from distutils.version import LooseVersion
if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
    GradientDescentOptimizer = tf.train.GradientDescentOptimizer
    AdamOptimizer = tf.train.AdamOptimizer
else:
    GradientDescentOptimizer = tf.optimizers.SGD
    AdamOptimizer = tf.optimizers.Adam



class auditing_CV:
    def load_data(self, data_dir):
        tst_x, tst_y = None, None
        if FLAGS.dataset.startswith("fmnist"):
            data_dir = os.path.join(data_dir, "fmnist")
            path = os.path.join(data_dir, f"clipbkd-new-{FLAGS.n_pois}.npy")
            (nobkd_trn_x, nobkd_trn_y), (bkd_trn_x, bkd_trn_y), _, _ = np.load(path, allow_pickle=True)
            bkd_trn_y = np.eye(2)[bkd_trn_y]
            nobkd_trn_y = np.eye(2)[nobkd_trn_y]
            bkd_x, bkd_y = None, None
        
        elif FLAGS.dataset.startswith('p100'):
            path = os.path.join(data_dir, os.path.join(FLAGS.dataset, 'p100_{}.npy'.format(FLAGS.n_pois)))
            (nobkd_trn_x, nobkd_trn_y), (bkd_trn_x, bkd_trn_y), (bkd_x, bkd_y), _ = np.load(path, allow_pickle=True)
            nobkd_trn_y = np.eye(100)[nobkd_trn_y]
            bkd_trn_y = np.eye(100)[bkd_trn_y]
            FLAGS.learning_rate = 2
            FLAGS.epochs = 100
        
        return bkd_trn_x, bkd_trn_y, nobkd_trn_x, nobkd_trn_y, bkd_x, bkd_y, tst_x, tst_y
    
    def build_model(self, x, y):
        input_shape = x.shape[1:]
        num_classes = y.shape[1]
        print(input_shape, num_classes)
        
        if FLAGS.dataset.startswith('fmnist'):
            l2_reg = 0
        elif FLAGS.dataset.startswith('p100'):
            if FLAGS.model == 'lr':
                l2_reg = 1e-5
            else:
                assert FLAGS.model == '2f'
                l2_reg = 1e-4
        
        if FLAGS.model == 'lr':
            model = tf.keras.Sequential([
                    tf.keras.layers.Flatten(input_shape=input_shape),
                    tf.keras.layers.Dense(num_classes, kernel_initializer='glorot_normal',
                        kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
                    ])
        elif FLAGS.model == '2f':
            model = tf.keras.Sequential([
                    tf.keras.layers.Flatten(input_shape=input_shape),
                    tf.keras.layers.Dense(32, activation='relu', kernel_initializer='glorot_normal',
                        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
                    tf.keras.layers.Dense(num_classes, kernel_initializer='glorot_normal',
                        kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
                    ])
        else:
            raise NotImplementedError
        
        return model
    
    def train_model(self, model, train_x, train_y, test_x, test_y, savepath, new_seed):
        tf.random.set_seed(new_seed)
        
        if FLAGS.noise_type == "lmo":
            print("Using the LMO settings.")
            print(f"check this value 1: {FLAGS.noise_params}")
            from lmo_config.lmo_args import LMO_ARGS
            print(LMO_ARGS[1])
            import lmo_config.optimizers as dp_optimizer_vectorized
            print(FLAGS.noise_params)
        elif FLAGS.noise_type == "gaussian":
            from tensorflow_privacy.privacy.optimizers import dp_optimizer_vectorized
        
        print('here')
        print(FLAGS.noise_params)
        optimizer = dp_optimizer_vectorized.VectorizedDPSGD(
            l2_norm_clip=FLAGS.l2_norm_clip,
            noise_multiplier=FLAGS.noise_params,
            num_microbatches=FLAGS.microbatches,
            learning_rate=FLAGS.learning_rate)
        
        loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True) # , reduction=tf.losses.Reduction.NONE)
        
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        print("not changing weights")
        
        np.random.seed(42)
        sample_fraction = 0.1
        sample_indices = np.random.choice(len(train_x), int(len(train_x) * sample_fraction), replace=False)
        train_x = train_x[sample_indices]
        train_y = train_y[sample_indices]
        
        model.fit(train_x, train_y,
                epochs=FLAGS.epochs,
                validation_data=(test_x, test_y),
                batch_size=FLAGS.batch_size)
        
        model.save(savepath)
        print(f"model is saved at {savepath}")


def main(_):
    dataset = FLAGS.dataset
    FLAGS.exp_name = FLAGS.dataset + "-" + FLAGS.model + "-" + FLAGS.noise_type + "-" + str(FLAGS.noise_params) + "-" + str(FLAGS.epochs) + "-" + str(FLAGS.l2_norm_clip)
    # save_dir = f"/mnt/nvme1n1p1/home/qiy22005/PRO/noise_mia/models"
    
    from init import init
    exp_type, data_dir, _, model, _, _, _, _, save_dir, _, _ = init(FLAGS.noise_type, FLAGS.dataset, FLAGS.model)
    
    
    assert "cv" in exp_type or "nlp" in exp_type
    suffix = '.h5' if "cv" in exp_type else '.safetensors'
    savepath = os.path.join(save_dir, FLAGS.exp_name+suffix)
    if os.path.exists(savepath):
        print(f"{savepath} exists.")
        exit(0)
    
    dataset = "p100"
    data_dir = "/root/noise_mia/noise_auditing/auditing/datasets"
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
        print(len(all_bkds["trn"][0]))
    
    
    trn_x, trn_y = all_bkds['trn']
    tst_x, tst_y = all_bkds['tst']
    
    auditor = auditing_CV()
    model = auditor.build_model(trn_x, trn_y)

    np.random.seed(None)
    new_seed = np.random.randint(1000000)
    auditor.train_model(model, trn_x, trn_y, trn_x, trn_y, savepath, new_seed)
    
    model = tf.keras.models.load_model(savepath)
    tst_preds = softmax(model.predict(tst_x), axis=1)
    
    predict_single_label = np.argmax(tst_preds, axis=1)
    true_single_label = np.argmax(tst_y, axis=1)
    accuracy = accuracy_score(true_single_label, predict_single_label)
    
    print(f"print_results: {FLAGS.exp_name} {accuracy}")



if __name__ == '__main__':
    app.run(main)