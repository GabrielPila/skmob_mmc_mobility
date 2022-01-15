import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import skmob
from tqdm import tqdm
import sys
sys.path.append('../')
from src.geo_utils import (get_clusters_from_tdf,
                            assign_tdf_points_to_clusters,
                            get_mmc_transitions,
                            get_stationary_vector)
import os
warnings.filterwarnings('ignore')
def get_distance(row):
    lat, lng, lat_last, lng_last = row['lat'], row['lng'], row['lat_last'], row['lng_last']
    return skmob.utils.utils.distance((lat, lng), (lat_last, lng_last))
    # Reading all the DataFrames and consolidating it into one
data_files = sorted([x for x in os.listdir('../data/') if 'geo_' in x])

data = pd.DataFrame()
for data_file in tqdm(data_files):
    data_i = pd.read_csv(f'../data/{data_file}')
    data = data.append(data_i)
data = data.reset_index(drop=True)

# Homogenizing the dataset format

geo_columns = ['user', 'hour', 'lat', 'lng']

data['hour'] = data['date'] + ' ' + data['time']
data['hour'] = pd.to_datetime(data['hour'])
data['user'] = data['user_id'].map(int)

data = data[geo_columns]
data = data.drop_duplicates()
data = data.groupby(['user','hour'], as_index=False).nth(0)
data = data.reset_index(drop=True)


import gc
gc.collect()

"""

# Estimating distance and time differences

data[['lat_last','lng_last']] = data.groupby('user').shift(1)[['lat','lng']]
data['seconds_diff'] = data.groupby('user')['hour'].diff(1).dt.seconds

data['distance_to_last_km'] = data.apply(get_distance, axis=1) 
data['speed_mps'] = data['distance_to_last_km'] / data['seconds_diff'] * 1000

del data['lat_last'], data['lng_last']
"""

# Saving Data
data.to_parquet('../data/geolife_consolidated.parquet', index=False)
