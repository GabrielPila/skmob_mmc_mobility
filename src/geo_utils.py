import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import skmob
from tqdm import tqdm
from skmob.preprocessing import (filtering, 
                                 detection, 
                                 compression, 
                                 clustering)


warnings.filterwarnings('ignore')


def get_clusters_from_tdf(tdf,
                          filter_noise=True,
                          max_speed_kmh = 50,
                          detect_stops=True,
                          minutes_for_a_stop = 20,
                          compress=True,
                          spatial_radius_km = 0.2,
                          spatial_radius_compress_km = 0.3,
                          cluster_radius_km = 1,
                          min_samples=1,
                          verbose=True):
    '''Get Clusters From TDF
    
    Generates clusters from a trajectory dataframe.
    
    Parameters:
    -----------
        tdf (Trajectory Data Frame): 
        max_speed_kmh (int):
        minutes_for_a_stop (int): 
        spatial_radius_km (float): 
        spatial_radius_compress_km (float)
        cluster_radius_km (float):
        verbose (bool): 
    
    Returns:
    --------
        clusters (Data Frame): The Dataframe of the clusters with lat and lng.
    '''
    if filter_noise:
        # 1. Noise Filtering
        tdf_f = filtering.filter(tdf, 
                                 max_speed_kmh=max_speed_kmh)
        if verbose: print('INFO: Noise Filtering applied')
    
    if detect_stops:
    # 2. Detection Stops
        tdf_fs = detection.stops(tdf_f, 
                                 minutes_for_a_stop=minutes_for_a_stop,
                                 spatial_radius_km=spatial_radius_km,
                                 leaving_time=True,
                                 min_speed_kmh=None)
        if verbose: print('INFO: Stops generated applied')
    else:
        tdf_fs = tdf_f
        
    # 3. Compression
    if compress:
        tdf_fsc = compression.compress(tdf_fs, 
                                       spatial_radius_km=spatial_radius_compress_km)
        if verbose: print('INFO: Stops compressed')
    else:
        tdf_fsc = tdf_fs            

    # 4. Clustering
    tdf_fsccl = clustering.cluster(tdf_fsc, 
                                   cluster_radius_km=cluster_radius_km,
                                   min_samples=1)
    if verbose: print('INFO: Clusters generated')

    print(tdf.shape, tdf_f.shape, tdf_fs.shape, tdf_fsc.shape, tdf_fsccl.shape)

    clusters = tdf_fsccl.groupby(['cluster'])[['lat','lng']].median().reset_index()
    print(f'INFO: {len(clusters)} clusters generated.')

    m = tdf_fsccl.plot_stops(zoom=11)
        
    return clusters, m



def assign_tdf_points_to_clusters(tdf, clusters, 
                                  max_radius_to_cluster_km=0.2):
    '''Assign TDF Points to Clusters
    
    Attempts to assign the corresponding cluster to each of the rows of the TDF.
    
    Parameters:
    -----------
        tdf (Trajectory Data Frame): tdf to be assigned.
        clusters (Data Frame): clusters to be assigned.
        max_radius_to_cluster_km (float): maximum distance to consider a point part of a cluster.
        
    Returns:
    --------
        tdf_ (Trajectory Data Frame): tdf with the clusters assigned (labelled).
        cluster_distances (Data Frame): distance from each point to each cluster.
    
    '''
    ########################## CLUSTER LABELLING #########################
    # Assign each point to a cluster (where possible)
    
    def get_distance_from_cluster(row, coord_cluster):
        coord_tdf = (row['lat'], row['lng'])
        return skmob.utils.utils.distance(coord_tdf, coord_cluster)
    
    tdf_ = tdf.copy()
    cluster_distances = pd.DataFrame(index=tdf_.index)
    for i, cluster in tqdm(list(clusters.iterrows())):
        cluster_coord = (cluster['lat'], cluster['lng'])
        cluster_distances[f'd_cl_{i:02d}'] = tdf_.apply(get_distance_from_cluster, axis=1, args=[cluster_coord])

    # We will not consider the distances higher than max_radius_to_cluster_km
    cluster_distances_1 = cluster_distances[(cluster_distances <= max_radius_to_cluster_km)]

    # We will assign the point to the closer cluster 
    tdf_['cluster'] = cluster_distances_1.idxmin(axis=1)
    return tdf_, cluster_distances



def get_mmc_transitions(tdf):
    '''Get MMC Transitions
    
    Returns the tdf with the transitions ocurred ammong clusters.
    
    Parameters:
    -----------
        tdf (Trajectory Data Frame): tdf with the clusters already assigned.
    
    Returns:
    --------
        transit_df (Trajectory Data Frame): tdf with different origin and end clusters.
    '''
    ##################### CLUSTER TRANSITIONS ####################
    mmc_df = tdf.dropna(subset=['cluster'])
    mmc_df['cluster_next'] = mmc_df['cluster'].shift(-1)

    mmc_df = mmc_df.dropna(subset=['cluster_next'])
    mmc_df['transition'] = mmc_df['cluster']+'-'+mmc_df['cluster_next']
    transit_df = mmc_df[mmc_df['cluster']!=mmc_df['cluster_next']]
    return transit_df


def get_stationary_vector(transitMatrix):
    '''Get Stationary Vector
    
    Returns the stationary vector from a transition Matrix
    
    Parameters
    ----------
        transitMatrix (np.array - shape=(2,2)): Transition Matrix
    
    Returns
    -------
        stationary_vector (np.array): Stationary Vector
    '''
    n = transitMatrix.shape[0]
    print('Shape of transitMatrix: ', transitMatrix.shape)
    A = np.append(transitMatrix.T - np.identity(n), np.ones(n).reshape((1,n)), axis=0)
    B = np.zeros(n+1).reshape((n+1,1))
    B[n][0]=1
    stationary_vector = np.linalg.solve((A.T).dot(A), (A.T).dot(B)) 
    return stationary_vector