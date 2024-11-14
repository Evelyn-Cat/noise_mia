import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import softmax

DEFAULT_MAX_ORDER = 128
DEFAULT_ORDER = range(2, DEFAULT_MAX_ORDER + 1)
DEFAULT_DISTRIBUTIONS = ["Gamma", "Exponential", "Uniform"]

try:
    import tensorflow as tf
    from distutils.version import LooseVersion
    if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
        GradientDescentOptimizer = tf.train.GradientDescentOptimizer
        AdamOptimizer = tf.train.AdamOptimizer
    else:
        GradientDescentOptimizer = tf.optimizers.SGD
        AdamOptimizer = tf.optimizers.Adam
except:
    pass

def load_txt(filepath, columns):
    with open(filepath, 'r', encoding='utf-8') as f:
        rs = f.readlines()

    df = pd.DataFrame([], columns=columns)
    for r in rs:
        name, value = r.strip().split(":")[0], r.strip().split(":")[1]
        df.loc[0, name] = value
    return df

# noise realted: load Qt.mat
def load_Qt_mat(filepath, columns):
    try:
        data = loadmat(filepath)
        Qt = data['Qt']
        df = pd.DataFrame(Qt, columns=columns)
    except:
        df = pd.read_csv(filepath)
        df = df[columns]
    return df


# noise realted: generate noise
def generate_noise(params, distributions=DEFAULT_DISTRIBUTIONS, noise_size=1):
    bs = generate_b(params, distributions=distributions, noise_size=noise_size)
    return np.random.laplace(0, bs)

def generate_b(params, distributions=DEFAULT_DISTRIBUTIONS, noise_size=1):
    us = 0
    
    if "Gamma" in distributions:
        us = us + params['a1']*np.random.gamma(params["G_k"], params["G_theta"], noise_size)
    else:
        us = us + 0
    
    if "Exponential" in distributions:
        us = us + params['a3']*np.random.exponential(params["E_lambda"], noise_size)
    else:
        us = us + 0
    
    if "Uniform" in distributions:
        us = us + params['a4'] * np.random.uniform(params["U_a"], params["U_b"], noise_size)
    else:
        us = us + 0
    
    return 1/us


# noise related functions
def compute_ma(N, order=1.1, sensitivity=1, distributions=["Gamma", "Exponential", "Uniform"], T=1):
    def compute_M(t, distributions, T=1):
        MGFs = 1
        if "Gamma" in distributions:
            try:
                MGF_Gamma = ((1-N['a1']*t*N['G_theta'])**(-N['G_k']))  # Gamma
            except:
                print(f"MGF_Gamma cannot be computed so we pass this option: {N}")
                return np.nan
            MGFs = MGFs * MGF_Gamma
        elif "Exponential" in distributions:
            try:
                MGF_Exp = (N['E_lambda']/(N['E_lambda']-N['a3']*t))  # Exponential
            except:
                print(f"MGF_Exp cannot be computed so we pass this option: {N}")
                return np.nan
            MGFs = MGFs * MGF_Exp
        elif "Uniform" in distributions:
            try:
                MGF_Uniform = ((np.exp(N['a4']*t*N['U_b'])-np.exp(N['a4']*t*N['U_a']))/(N['a4']*t*(N['U_b']-N['U_a'])))  # Uniform
            except:
                print(f"MGF_Uniform cannot be computed so we pass this option: {N}")
                return np.nan
            MGFs = MGFs * MGF_Uniform
        
        MGFs = MGFs ** T
        return MGFs
    
    try:
        MGF1 = compute_M(t=order*sensitivity, distributions=distributions, T=T)
        MGF2 = compute_M(t=-(order+1)*sensitivity, distributions=distributions, T=T)
        ma_N = np.log(((order+1) * MGF1 + order * MGF2)/(2*order+1))
        print("MGF1, MGF2 or ma_N cannot be computed.")
    except:
        return np.nan
    
    return ma_N


def theoretical_mia(N, sensitivity=1, epsilon=1, alpha=0.2, distributions=["Gamma", "Exponential", "Uniform"], T=1):
    betas = {}
    for order in DEFAULT_ORDER:
        try:
            ma = compute_ma(N, order=order, sensitivity=sensitivity, distributions=distributions, T=T)
            if ma == np.nan:
                return [], [], [], [], [], []

            delta = np.exp(ma - order * epsilon)
            beta1 = 1 - delta - np.exp(epsilon) * alpha
            beta2 = np.exp(-epsilon) * (1 - delta - alpha)
            
            if not np.isnan(beta1) and not np.isnan(beta2):
                betas[order] = np.max([0, beta1, beta2])
            elif not np.isnan(beta1) and np.isnan(beta2):
                betas[order] = np.max([0, beta1])
            elif np.isnan(beta1) and not np.isnan(beta2):
                betas[order] = np.max([0, beta2])
        except:
            continue
    
    if betas:
        beta_index = max(betas, key=betas.get)
        beta = betas[beta_index]

        mia = 0.5 * (1 - alpha - beta)
        
        return betas, beta_index, beta, mia, epsilon, delta
    else:
        return [], [], [], [], [], []



# task related: preprocess
def preprocess_df(df, remove_nan=True, remove_inf=True):
    if remove_nan and remove_inf:
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
    if remove_nan and not remove_inf:
        df = df.dropna()
    if not remove_nan and remove_inf:
        df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
    return df


# task related: load dataset
def read_dataset(dataset, data_dir):
    tasktype = "cv" if dataset=="p100" or dataset=="fmnist" else "nlp"

    if "fmnist" in dataset:
        loader = np.load(os.path.join(data_dir, "fmnist/clipbkd-new-1.npy"), allow_pickle=True)
        data = {"tst": loader[3],"trn": loader[0]}
        data["tst"] = data["tst"][0].reshape((-1, 28, 28, 1)), np.eye(2)[data["tst"][1]]
        data["trn"] = data["trn"][0].reshape((-1, 28, 28, 1)), np.eye(2)[data["trn"][1]]
        learning_rate = 0.15
    elif "p100" in dataset:
        loader = np.load(os.path.join(data_dir, "p100/p100_1.npy"), allow_pickle=True)
        data = {"tst": loader[3],"trn": loader[0]}
        data["tst"] = data["tst"][0].reshape((-1, 100)), np.eye(100)[data["tst"][1]]
        data["trn"] = data["trn"][0].reshape((-1, 100)), np.eye(100)[data["trn"][1]]
        print(len(data["trn"][0]))
        learning_rate = 2
    return tasktype, data, learning_rate


# task related: mia attack evaluation
def mia(model, trn_x, trn_y, tst_x, tst_y):
    np.random.seed(0)
    tst_y_len = tst_y.shape[0]
    trn_y_inds = np.random.choice(trn_y.shape[0], tst_y_len, replace=False)
    trn_x, trn_y = trn_x[trn_y_inds], trn_y[trn_y_inds]
    
    trn_preds = softmax(model.predict(trn_x), axis=1)
    tst_preds = softmax(model.predict(tst_x), axis=1)
    trn_loss = np.multiply(trn_preds, trn_y).sum(axis=1)
    tst_loss = np.multiply(tst_preds, tst_y).sum(axis=1)
    
    trn_loss_mean = trn_loss.mean()
    trn_thresh = (trn_preds >= trn_loss_mean).sum()
    tst_thresh = tst_y_len - (tst_preds >= trn_loss_mean).sum()
    acc = (trn_thresh + tst_thresh) / tst_y_len
    return np.log(acc)


# task related: CV task relate
class mia_acc_CV:
    def __init__(self, dataset, model, epochs, microbatches, learning_rate, batchsize, clip, noise_type, noise_params):
        self.dataset=dataset
        self.model=model
        self.microbatches=microbatches
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.batchsize=batchsize
        self.l2_norm_clip=clip
        self.noise_type=noise_type
        self.noise_params=noise_params

    def build_model(self, x, y):
        input_shape = x.shape[1:]
        num_classes = y.shape[1]
        print(input_shape, num_classes)
        
        if self.dataset.startswith('fmnist'):
            self.l2_reg = 0
        elif self.dataset.startswith('p100'):
            if self.model == 'lr':
                self.l2_reg = 1e-5
            else:
                assert self.model == '2f'
                self.l2_reg = 1e-4
        
        if self.model == 'lr':
            model = tf.keras.Sequential([
                    tf.keras.layers.Flatten(input_shape=input_shape),
                    tf.keras.layers.Dense(num_classes, kernel_initializer='glorot_normal',
                        kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))
                    ])
        elif self.model == '2f':
            model = tf.keras.Sequential([
                    tf.keras.layers.Flatten(input_shape=input_shape),
                    tf.keras.layers.Dense(32, activation='relu', kernel_initializer='glorot_normal',
                        kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)),
                    tf.keras.layers.Dense(num_classes, kernel_initializer='glorot_normal',
                        kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))
                    ])
        else:
            raise NotImplementedError
        
        return model
    
    def train_model(self, model, train_x, train_y, test_x, test_y, savepath, new_seed):
        tf.random.set_seed(new_seed)
        
        if self.noise_type == "mg":
            import optimizers as dp_optimizer_vectorized
        elif self.noise_type == "gaussian":
            from tensorflow_privacy.privacy.optimizers import dp_optimizer_vectorized
        
        optimizer = dp_optimizer_vectorized.VectorizedDPSGD(
            l2_norm_clip=self.l2_norm_clip,
            noise_multiplier=self.noise_params,
            num_microbatches=self.microbatches,
            learning_rate=self.learning_rate
        )
        
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        
        model.fit(train_x, train_y,
                epochs=self.epochs,
                validation_data=(test_x, test_y),
                batch_size=self.batchsize)
        
        model.save(savepath)
        print(f"model is saved at {savepath}")



