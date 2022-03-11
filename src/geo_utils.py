import os
import datetime
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skmob
from skmob.preprocessing import (filtering, 
                                 detection, 
                                 compression, 
                                 clustering)
from tqdm import tqdm


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
    #filter_noise = False
    if filter_noise:
        # 1. Noise Filtering
        tdf_f = filtering.filter(tdf, 
                                 max_speed_kmh=max_speed_kmh)
        if verbose: print('INFO: Noise Filtering applied')
    else:
        tdf_f = tdf
    
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
    mmc_df = tdf[tdf['cluster'].notnull()]
    mmc_df['cluster_next'] = mmc_df['cluster'].shift(-1)

    mmc_df = mmc_df[mmc_df['cluster_next'].notnull()]
    #mmc_df.loc[mmc_df['cluster_next'].isnull(), 'cluster_next'] = mmc_df.iloc[1]['cluster']
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



# Distance Estimation

def get_distance_bw_clusters(cluster_1, cluster_2):
    '''Get Distance Between Clusters
    
    Parameters
    ----------
        cluster_1 (pd.Dataframe): df with columns ['cluster', 'lat', 'lng', 'sta_vector']
        cluster_2 (pd.Dataframe): df with columns ['cluster', 'lat', 'lng', 'sta_vector']
    
    Returns
    -------
        distance (float): Distance between clusters (one-way)
    '''
    valid_distances = []
    valid_clusters = []
    for i, row_i in cluster_1.iterrows():    
        coord_i = (row_i['lat'], row_i['lng'])

        clusters, distances = [], []    
        for j, row_j in cluster_2.iterrows():
            cluster_j = row_j['cluster']
            coord_j = (row_j['lat'], row_j['lng'])

            distance_ij = skmob.utils.utils.distance(coord_i, coord_j)    
            clusters.append(cluster_j)
            distances.append(distance_ij)

        distances = np.array(distances)

        idx_min = distances.argmin()
        min_dist = distances.min()
        cluster_min = clusters[idx_min]

        valid_distances.append(min_dist)
        valid_clusters.append(cluster_min)

    cluster_est = cluster_1.copy()
    cluster_est['cluster_other'] = valid_clusters
    cluster_est['distance_other'] = valid_distances
    distance = (cluster_est['distance_other'] * cluster_est['sta_vector']).sum()
    return distance


def get_mean_distance_bw_clusters(cluster_1, cluster_2):
    '''Get Mean Distance Between Clusters
    
    Get the distances of cluster_1 to cluster_2 and viceversa and returns the average.

    Parameters
    ----------
        cluster_1 (pd.Dataframe): df with columns ['cluster', 'lat', 'lng', 'sta_vector']
        cluster_2 (pd.Dataframe): df with columns ['cluster', 'lat', 'lng', 'sta_vector']
    
    Returns
    -------
        mean_distance (float): Distance between clusters (two-way)
    '''
    d1 = get_distance_bw_clusters(cluster_1, cluster_2)
    d2 = get_distance_bw_clusters(cluster_2, cluster_1)
    mean_distance = (d1+d2)/2
    return mean_distance


def get_mmc_distances_matrix(mmc_list_a, mmc_list_b):
    '''Get MMC Distances Matrix
    
    It estimates the distances of the elements of two lists of MMCs.
    Afterwards, it stores the distances in a matrix.
    
    Parameters
    ----------
        mmc_list_a (list): List of MMCs A
        mmc_list_b (list): List of MMCs B

    Returns
    -------
        distance_matrix (np.array): Matrix with all the distances calculated.
    
    '''
    n = len(mmc_list_a)
    m = len(mmc_list_b)
    distance_matrix = np.zeros((n, m))
    for i, mmc_i in enumerate(mmc_list_a):
        for j, mmc_j in enumerate(mmc_list_b):
            distance = get_mean_distance_bw_clusters(mmc_i, mmc_j)
            distance_matrix[i,j] = round(distance,4)
            
    return distance_matrix


############ ALL COMBINED ##############

def get_mmc_clusters_stavectors(geo):
    '''Get MMC Clusters and Stationary Vectors
    
    Parameters
    ----------
        geo (pd.Dataframe): Dataframe of 1 user_id. Must contain only the columns: ['user', 'hour', 'lat', 'lng']
    
    Returns
    -------
        clusters (pd.Dataframe): Dataframe with the clusters and the stationary value
        m (folium.folium.Map): Map generated with the clusters on it.
    '''
    # TDF Definition
    trgeo = skmob.TrajDataFrame(
        geo, 
        datetime='hour',
        user_id='user'
    )

    # Cluster Generation
    clusters, m = get_clusters_from_tdf(
        trgeo,
        verbose=True,
        max_speed_kmh= 0.1,
        detect_stops=False,
        compress=False ,
        minutes_for_a_stop=2,
        spatial_radius_km=0.2,
        spatial_radius_compress_km=.2,
        cluster_radius_km=0.5,
        min_samples=2
    )

    # Cluster Assignation
    trgeo_cl, distances = assign_tdf_points_to_clusters(
        tdf=trgeo, 
        clusters=clusters
    )

    # Generation of Transit Dataframe
    transit_df = get_mmc_transitions(trgeo_cl)

    # Generation of Transit Matrix
    transit_matrix = pd.crosstab(transit_df['cluster'], 
                                 transit_df['cluster_next'],
                                 normalize='index').values
    print(transit_matrix)
    
################################## START FIX ANTHONY ################################

#Lógica adicional para añadir entradas o salidas y hacer simétrica la matriz
    
    #Registros cuyos clusters nunca aparecen como salida
    cluster_mode = transit_df['cluster_next'].mode()[0] # Cluster más frecuente
    df_never_next= transit_df[~transit_df['cluster'].isin(transit_df['cluster_next'].unique())]
    df_never_next['cluster_next'] = df_never_next['cluster']
    df_never_next['cluster'] = cluster_mode
    if(df_never_next.shape[0]>0):
        print('\nMostrando registros ficticios ')
        df_never_next['datetime'] = df_never_next['datetime']  - datetime.timedelta(milliseconds = 10)
        print(df_never_next)
        
        transit_df = pd.concat([transit_df,df_never_next], ignore_index=True ).sort_values(by='datetime')
        
    #Registros cuyos clusters nunca aparecen como inicio
    df_never_ini= transit_df[~transit_df['cluster_next'].isin(transit_df['cluster'].unique())]
    df_never_ini['cluster'] = df_never_ini['cluster_next']
    df_never_ini['cluster_next'] = cluster_mode
    if(df_never_ini.shape[0]>0):
        print('\nMostrando registros ficticios')
        df_never_ini['datetime'] = df_never_ini['datetime']  + datetime.timedelta(milliseconds = 10)
        
        print(df_never_ini)
        transit_df = pd.concat([transit_df,df_never_ini], ignore_index=True ).sort_values(by='datetime')
    
    transit_matrix = pd.crosstab(transit_df['cluster'], 
                                 transit_df['cluster_next'],
                                 normalize='index')
    print(transit_matrix)

    print('Getting stationary Vector...')
    stat_vec = get_stationary_vector(transit_matrix)
    print('Stationary Vector generated:')
    print(stat_vec)
    
    return clusters, m, transit_matrix, transit_df, stat_vec




def analyze_time_drift(data):
    '''Analiza la distribución de la variación de tiempo (segundos) entre evento y evento'''
    
    # Distribución de número de eventos
    serie_seconds_events = data['seconds_diff'].value_counts()
    serie_seconds_events_log = np.log10(serie_seconds_events)

    print(serie_seconds_events)
    print()
    print(serie_seconds_events.quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]))

    fig = plt.figure(figsize=(15,4))
    fig.add_subplot(1,2,1)
    data['seconds_diff'].hist(bins=100)
    plt.ylabel('Numero de Eventos')
    plt.xlabel('Delta de Segundos')
    
    fig.add_subplot(1,2,2)

    serie_seconds_events_log.hist(bins=30)
    plt.title('Distribución de eventos según drift en segundos (log10)');
    plt.ylabel('Numero de Eventos')
    plt.xlabel('Delta de Segundos (log10 scale)');
    

def analyze_space_drift(data):
    '''Analiza la distribución de la variación de distancia (Km) entre evento y evento'''

    serie_distance_events = data['distance_to_last_km'].value_counts()
    serie_distance_events_log = np.log10(serie_distance_events)
    
    print(serie_distance_events)
    print()
    print(serie_distance_events.quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]))

    fig = plt.figure(figsize=(15,4))
    fig.add_subplot(1,2,1)
    data['distance_to_last_km'].hist(bins=100)
    plt.ylabel('Numero de Eventos')
    plt.xlabel('Delta de KM');
    
    fig.add_subplot(1,2,2)

    serie_distance_events_log.hist(bins=30)
    plt.title('Distribución de eventos según drift de distancia (log10)');
    plt.ylabel('Numero de Eventos')
    plt.xlabel('Delta de KM (log10 scale)');
    
    
def analyze_speed_drift(data):
    '''Analiza la velocidad (metros por segundo) de cada evento'''

    serie_speed = data['speed_mps']
    serie_speed_log = np.log10(serie_speed+1)
    
    print(serie_speed.quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]))

    fig = plt.figure(figsize=(15,4))
    
    fig.add_subplot(1,2,1)
    serie_speed.hist(bins=100)
    plt.title('Distribución de eventos según velocidad');
    plt.ylabel('Numero de Eventos')
    plt.xlabel('Velocidad');
    
    fig.add_subplot(1,2,2)
    serie_speed_log.hist(bins=30)
    plt.title('Distribución de eventos según velocidad (log10)');
    plt.ylabel('Numero de Eventos')
    plt.xlabel('Velocidad (log10 scale)');
    
    
def analyze_number_of_events_per_day(data):
    '''Analiza la distribución de la variación de tiempo (segundos) entre evento y evento'''

    # Distribución de número de eventos
    df_events_per_day = data.groupby(['date','user'], as_index=False).size()
    
    serie_user_events_per_day = df_events_per_day['size']
    serie_user_events_per_day_log = np.log10(serie_user_events_per_day)

    print(serie_user_events_per_day)
    print()
    print(df_events_per_day['size'].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]))

    fig = plt.figure(figsize=(15,4))

    fig.add_subplot(1,2,1)
    serie_user_events_per_day.hist(bins=30)
    plt.title('Distribución de número de eventos por usuarios por día');
    plt.ylabel('Numero de Usuarios')
    plt.xlabel('Número de eventos en un día')    
    
    fig.add_subplot(1,2,2)
    serie_user_events_per_day_log.hist(bins=30)
    plt.title('Distribución de número de eventos por usuarios por día (log10)');
    plt.ylabel('Numero de Usuarios')
    plt.xlabel('Número de eventos en un día (log10 scale)')
    
def analyze_number_of_events(data):
    # Distribución de número de eventos
    serie_user_events = data['user'].value_counts()
    serie_user_events_log = np.log10(serie_user_events)

    print(serie_user_events)
    print()
    print(serie_user_events.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]))

    fig = plt.figure(figsize=(15,4))

    fig.add_subplot(1,2,1)
    serie_user_events.hist(bins=30)
    plt.title('Distribución de usuarios según Nro de Eventos');
    plt.ylabel('Numero de Usuarios')
    plt.xlabel('Número de eventos')    
    
    fig.add_subplot(1,2,2)
    serie_user_events_log.hist(bins=30)
    plt.title('Distribución de usuarios según Nro de Eventos (log10)');
    plt.ylabel('Numero de Usuarios')
    plt.xlabel('Número de eventos (log10 scale)')
    

def analyze_dates_with_events(data):
    # Distribución de número de eventos
    serie_user_events = data[['user','date']].drop_duplicates()['user'].value_counts()
    serie_user_events_log = np.log10(serie_user_events)

    print(serie_user_events)

    fig = plt.figure(figsize=(15,4))
    fig.add_subplot(1,2,1)
    serie_user_events.hist(bins=30)
    plt.title('Distribución de usuarios según Nro de Días con Eventos');
    plt.ylabel('Numero de Usuarios')
    plt.xlabel('Número de Días con eventos')
    
    fig.add_subplot(1,2,2)

    serie_user_events_log.hist(bins=30)
    plt.title('Distribución de usuarios según Nro de Días con Eventos (log10)');
    plt.ylabel('Numero de Usuarios')
    plt.xlabel('Número de eventos (log10 scale)')

def filter_min_events_per_user(_df, min_events, user_col = 'user'):
    # Filtro de cantidad mínima de eventos por usuario
    print('Datos iniciales:')
    print('# Registros: {:,}'.format(_df.shape[0]), '\t|', '# Usuarios: {}'.format(_df[user_col].nunique()))
    print('-'*70)
    users_to_del = _df.groupby(user_col)[user_col].count()[_df.groupby(user_col)[user_col].count()<min_events].index
    _df = _df[~_df[user_col].isin(users_to_del)]
    
    print('Datos finales:')
    print('# Registros: {:,}'.format(_df.shape[0]), '\t|', '# Usuarios: {}'.format(_df[user_col].nunique()))
    print('-'*70)
    return _df      

def filter_min_days_per_user(_df, min_days, user_col = 'user', date_col  = 'date'):
    # Filtro de cantidad mínima de días reportados por usuario
    print('Datos iniciales:')
    print('# Registros: {:,}'.format(_df.shape[0]), '\t|', '# Usuarios: {}'.format(_df[user_col].nunique()))
    print('-'*70)
    users_to_del = _df.groupby(user_col)[date_col].nunique()[_df.groupby(user_col)[date_col].nunique()<min_days].index
    _df = _df[~_df[user_col].isin(users_to_del)]
    
    print('Datos finales:')
    print('# Registros: {:,}'.format(_df.shape[0]), '\t|', '# Usuarios: {}'.format(_df[user_col].nunique()))
    print('-'*70)
    return _df     