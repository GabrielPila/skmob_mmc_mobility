import numpy as np

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
                            get_stationary_vector,
                            get_distance_bw_clusters,
                            get_mean_distance_bw_clusters,
                            get_mmc_distances_matrix,
                            get_mmc_clusters_stavectors
                            )



# Reading 10 frames
data = pd.read_csv('../data/geo_000_009.csv.zip')

geo_columns = ['user', 'hour', 'lat', 'lng']
data['hour'] = data['date'] + ' ' + data['time']
data['user'] = data['user_id'].map(int)
data = data[geo_columns]

print()
print()
print(dir(skmob.preprocessing))
print()
print()

# Extraction of clusters 
geo_clusters = {}
geo_clusters_img = {}
geo_clusters_transit = {}
geo_clusters_transit_df = {}
users = sorted(data['user'].unique())
for user in tqdm(users):
    print(user)
#    try:

    geo = data[data['user']==user].reset_index(drop=True).copy()

    print(geo.shape)

    clusters, m, transit_matrix, transit_df = get_mmc_clusters_stavectors(geo)

    print(clusters)

    geo_clusters[user] = clusters
    geo_clusters_img[user] = m
    geo_clusters_transit[user] = transit_matrix
    geo_clusters_transit_df[user] = transit_df
#    except:
#        pass