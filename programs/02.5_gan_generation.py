import os
import time
from matplotlib.pyplot import title
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from config import PATH_LOCAL_DATA
import warnings 

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
    nepochs:int = 20,
    param:dict = {'batch_size': 64,
                'discriminatorDims': [64, 32, 16, 1],
                'generatorDims': [512, 3],
                'input_dim': 3,
                'optimizer': 'Adam',
                'random_dim': 100
                }
):

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

    data_conjoint, user_conjoint = get_data_user_conjoined(data)

    plot_user_geodata(data, user=user_conjoint, title='original', img_path=path_img)

    data_scaled, scaler = get_scaled_data(data_conjoint)

    dataset = tf.data.Dataset.from_tensor_slices(data_scaled).shuffle(50000).batch(param["batch_size"], drop_remainder=True)

    results = dp.train(dataset, nepochs, param["batch_size"], data.shape[0])

    gen_data = get_generated_data(results, scaler, user=user_conjoint)

    file_path_output = os.path.join(path_output, f'epochs_{nepochs}_{filename}')
    gen_data.to_csv(file_path_output, index=False)
    plot_user_geodata(gen_data, user=user_conjoint, title=f'generated_epoch_{nepochs}', img_path=path_img)


if __name__ == '__main__':
    train_gan()