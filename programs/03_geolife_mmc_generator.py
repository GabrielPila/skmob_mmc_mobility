import warnings
import sys
import re
import os
import datetime

import numpy as np
import pandas as pd
import skmob
from tqdm import tqdm
import pickle
import argparse

sys.path.append('../')

from src.geo_utils import (filter_min_events_per_user, 
                            filter_min_days_per_user,
                            get_mmc_clusters_stavectors)

print(os.getcwd())
print(__name__)
project_path='/home/anthony/projects/skmob_mmc_mobility/'


def main(
    path_geo_data: str= 'data/',
    path_mmc: str='experiment/',
    prefix: str=''
):
    print('starting main')
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type = str, help = "The name of the file saved in './data' folder to be processed" )
    parser.add_argument('--version', type = int, help = "specify the version of the experiment. By default is without version")

    args = parser.parse_args()


    # Reading data
    filename = args.filename
    fullpath =  os.path.join(project_path,path_geo_data,args.filename) #'geo_000_009.csv.zip'

    print(f'Reading the file at {fullpath}')



    if re.search('.csv',filename):
        df = pd.read_csv(fullpath)
    elif re.search('.parquet',filename):
        df = pd.read_parquet(fullpath)
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
    filename_prefix = filename.split('.')[0]
    if args.version:
        output_path = os.path.join(project_path,path_mmc,filename_prefix,'v'+str(args.version))
    else:
        output_path = os.path.join(project_path,path_mmc,filename_prefix,'release')

    os.makedirs(os.path.dirname(output_path), exist_ok=True) #version folder
    os.makedirs(os.path.dirname(os.path.join(output_path,'logs/')), exist_ok=True) #log inside

    for val in ['geo_clusters','geo_clusters_transit','geo_clusters_transit_df','stat_vectors']:
        if val == 'stat_vectors':
            output_fullpath = os.path.join(output_path,val)
        else:
            output_fullpath = os.path.join(output_path,'logs',val)
        
        with open(output_fullpath, 'wb') as f:
            pickle.dump(eval(val),f)
        print(f'{output_fullpath} saved...')

if(__name__=='__main__'):
    main()

