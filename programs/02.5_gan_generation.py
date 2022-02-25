import os
import time
from matplotlib.pyplot import title
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from config import PATH_LOCAL_DATA
import warnings 
from scipy.stats import ks_2samp
import json

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(tf.__version__)
print(gpus)

from gan_utils.dpgan_tf2 import DPGAN
from gan_utils.gan_tf2 import GAN
from gan_utils.gan_utils import (get_optimizers, 
                                get_scaled_data, 
                                get_generated_data, 
                                get_data_user_conjoined, 
                                plot_user_geodata)
warnings.filterwarnings('ignore')

def train_gan(
    path_data:str = os.path.join(PATH_LOCAL_DATA, 'users'),
    path_output:str = os.path.join(PATH_LOCAL_DATA, 'users_gan'),
    path_img:str = os.path.join(PATH_LOCAL_DATA, 'img'),
    filename:str = 'data_user_100.csv',
    nepochs:int = 2,
    param:dict = {'batch_size': 64,
                'discriminatorDims': [64, 32, 16, 1],
                'generatorDims': [512, 3],
                'input_dim': 3,
                'optimizer': 'Adam',
                'random_dim': 100
                }
):

    start_time = time.time()

    for path in [path_data, path_output, path_img]:
        if not os.path.exists(path):
            os.mkdir(path)

    file_path = os.path.join(path_data, filename)

    data = pd.read_csv(file_path)

    # Training
    g_optimizer, d_optimizer = get_optimizers(param["optimizer"])
        
    dp = GAN(
        param["input_dim"],
        param["random_dim"],
        param["discriminatorDims"],
        param["generatorDims"],
        g_optimizer,
        d_optimizer
    )
    d_dims = '_'.join([str(x) for x in param["discriminatorDims"]])
    g_dims = '_'.join([str(x) for x in param["generatorDims"]])

    data_conjoint, user_conjoint = get_data_user_conjoined(data)

    data_scaled, scaler = get_scaled_data(data_conjoint)
    dataset = tf.data.Dataset.from_tensor_slices(data_scaled).shuffle(50000).batch(param["batch_size"], drop_remainder=True)

    results = dp.train(dataset, nepochs, param["batch_size"], data.shape[0])
    gen_data = get_generated_data(results, scaler, user=user_conjoint)



    # Save Experiment
    exp_dir = f'user_{user_conjoint}_ddims_{d_dims}_gdims_{d_dims}_epochs_{nepochs}'
    execution_time = time.time() - start_time

    path_exp = os.path.join(path_output, exp_dir)
    if not os.path.exists(path_exp):
        os.mkdir(path_exp)

    plot_user_geodata(data, user=user_conjoint, title='original', img_path=path_exp)
    plot_user_geodata(gen_data, user=user_conjoint, title=f'generated_epoch_{nepochs}', img_path=path_exp)

    lat_metrics = ks_2samp(data_conjoint['lat'], gen_data['lat'])
    lon_metrics = ks_2samp(data_conjoint['lon'], gen_data['lon'])
    time_metrics = ks_2samp(pd.to_datetime(data_conjoint['time']).astype(int)/10**9, 
                        pd.to_datetime(gen_data['time']).astype(int)/10**9)

    dp.save_models(path=path_exp)
    dp.plot_loss_progress(path=path_exp)
    file_path_output = os.path.join(path_exp, f'gen_{filename}')
    gen_data.to_csv(file_path_output, index=False)


    registry_info = {
        'user_conjoint': user_conjoint,
        'exp_dir': exp_dir,
        'g_dims': g_dims,
        'd_dims': d_dims,
        'nepochs': nepochs,
        'ks_lat': lat_metrics[0],
        'ks_lon': lon_metrics[0],
        'ks_time': time_metrics[0],
        'ks_pv_lat': lat_metrics[1],
        'ks_pv_lon': lon_metrics[1],
        'ks_pv_time': time_metrics[1],
        'execution_time': execution_time
    }
    json.dump(registry_info, open(os.path.join(path_exp, 'registry_info.json'), 'w'))


if __name__ == '__main__':
    train_gan()