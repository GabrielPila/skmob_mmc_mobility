import numpy as np
import pandas as pd
import warnings
import skmob
from tqdm import tqdm
import sys
import re
import os
import datetime
import pickle
print(os.getcwd())
sys.path.append('../')
from src.geo_utils import (filter_min_events_per_user, 
                            filter_min_days_per_user,
                            get_mmc_clusters_stavectors)


# Reading data
filename = 'geo_000_009.csv.zip'
filename_prefix = filename.split('.')[0]

if re.search('.csv',filename):
    df = pd.read_csv('../data/' + filename)
elif re.search('.parquet',filename):
    df = pd.read_parquet('./data/' + filename)
else:
    raise Exception("Invalidad format provided")

# Show sample of data
print('\nDisplaying sample of records')
print(df.sample(10))
df.columns = list(df.columns)

#Temporal
df.rename(columns = {'user_id':'user'}, inplace=True)

# Creating required fields if not in dataset
if 'date' not in df.columns:
    df['date'] = df['hour'].dt.date

if 'datetime' not in df.columns:
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'] , format='%Y-%m-%d %H:%M:%S') 

# Show sample of data after columns created
print('\nDisplaying sample of records')
print(df.sample(10))

# Filtering based on number of events
df = filter_min_events_per_user(df, min_events=1000)

# Filtering based on number of days
df= filter_min_days_per_user(df, min_days=10)

# Extraction of clusters 
geo_clusters = {}
geo_clusters_img = {}
geo_clusters_transit = {}
geo_clusters_transit_df = {}
stat_vectors = {}
users = sorted(df['user'].unique()[:1])
for user in tqdm(users):
    print('Procesando user {}'.format(user))
    try: 
        geo = df[df['user']==user].reset_index(drop=True).copy()

        clusters, m, transit_matrix, transit_df, stat_vec = get_mmc_clusters_stavectors(geo)

        geo_clusters[user] = clusters
        geo_clusters_img[user] = m
        geo_clusters_transit[user] = transit_matrix
        geo_clusters_transit_df[user] = transit_df
        stat_vectors[user] = stat_vec
        print('Procesamiento del user {} TERMINADO...'.format(user))
        print('-'*100 + '\n')
    except:
        print('Error en la generación de clusters del user {}'.format(user))


# Save data

date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

for val in ['geo_clusters','geo_clusters_transit','geo_clusters_transit_df','stat_vectors']:
    tmp_filename = f'{filename_prefix}_{date}_{val}'
    with open(f'../data/{tmp_filename}', 'wb') as f:
        pickle.dump(eval(val),f)
    print(f'{tmp_filename} saved...')



