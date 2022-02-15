import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import time 
import os


def get_optimizers(optimizer='Adam'):
    if optimizer == "Adam":
        g_optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-4,
            beta_1=0.5,
            beta_2=0.9
        )
        d_optimizer = tf.keras.optimizers.Adam(
            learning_rate=1e-4,
            beta_1=0.5,
            beta_2=0.9
        )
    else:
        g_optimizer = tf.keras.optimizers.RMSprop(learning_rate=2.5e-5)
        d_optimizer = tf.keras.optimizers.RMSprop(learning_rate=2.5e-5)
        
    return g_optimizer, d_optimizer


def get_scaled_data(data):
    data_to_scale = data[['time','lat','lon']].copy()
    data_to_scale["time-f"] = pd.to_datetime(data_to_scale['time']).astype(int)/10**9
    data_to_scale = data_to_scale[['time-f','lat','lon']]
    scaler = StandardScaler()
    scaler.fit(data_to_scale)
    data_scaled = scaler.transform(data_to_scale)
    return data_scaled, scaler


def get_generated_data(results, scaler, user):
    temp = scaler.inverse_transform(results)
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