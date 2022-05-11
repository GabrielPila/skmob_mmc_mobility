import json
import os
import time 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tqdm import tqdm



def get_optimizers(
    optimizer='Adam',
    g_learning_rate = 1e-3, #Parametrized learning rate for generator
    d_learning_rate = 1e-3 #Parametrized learning rate for discriminator
    ):
    if optimizer == "Adam":
        g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=g_learning_rate,
            beta_1=0.5,
            beta_2=0.9
        )
        d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=d_learning_rate,
            beta_1=0.5,
            beta_2=0.9
        )
    else:
        g_optimizer = tf.keras.optimizers.RMSprop(learning_rate=2.5e-5)
        d_optimizer = tf.keras.optimizers.RMSprop(learning_rate=2.5e-5)
        
    return g_optimizer, d_optimizer


def get_data(data):
    data_to_scale = data[['time','lat','lon']].copy()
    data_to_scale["time-f"] = pd.to_datetime(data_to_scale['time']).dt.hour
    data_to_scale['lat_sqrd'] = data_to_scale['lat']**2
    data_to_scale['lon_sqrd'] = data_to_scale['lon']**2
    data_to_scale['lat_lon_sqrd'] = data_to_scale['lat']**2 + data_to_scale['lon']**2
    data_to_scale = data_to_scale[['time-f','lat','lon','lat_sqrd','lon_sqrd','lat_lon_sqrd']]

    return data_to_scale


def get_generated_data(results, scaler, user, scale_data):

    ## apply inverse transforme if necessary
    if scale_data: 
        temp = scaler.inverse_transform(results)
    else:
        temp = results.copy()
        
    temp = pd.DataFrame(temp)
    temp.rename(columns={1:'lat',
                            2:'lon',
                            0:'time'}, inplace=True)

    temp['time'] = pd.to_datetime(temp['time'], unit='s')
    temp['user'] = user
    temp = temp[['user','time','lat','lon']]
    return temp    


def get_data_user_conjoined(data):
    data_conjoint = data.copy()
    user_conjoint = '_'.join([str(x) for x in sorted(data['user'].unique())])
    data_conjoint['user'] = user_conjoint
    return data_conjoint, user_conjoint    


def plot_user_geodata(data, user, title='original', figsize=(15, 4), img_path='./'):
    fig = plt.figure(figsize=figsize)

    fig.add_subplot(1, 3, 1)
    sns.scatterplot(data['lat'],data['lon'])
    plt.title(f'Data {title} Lat-Lon - User {user}')

    fig.add_subplot(1, 3, 2)
    sns.distplot(data['lat'])
    plt.title(f'Data {title} Latitud - User {user}')

    fig.add_subplot(1, 3, 3)
    sns.distplot(data['lon'])
    plt.title(f'Data {title} Longitude - User {user}')

    plt.tight_layout();

    fig.savefig(os.path.join(img_path, f'{title}_user_{user}.png'))