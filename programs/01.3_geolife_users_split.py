import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

from config import PATH_LOCAL_DATA


def clean_data_geopoints(data, clean_weekends=False, clean_outliers=False, ndesv=2):
    if clean_weekends:
        data['day_name'] = pd.to_datetime(data['time']).dt.day_name()
        data = data[~(data['day_name'].isin(['Saturday','Sunday']))]
        del data['day_name']
        
    if clean_outliers:
        # Determine limits
        bounds_lat = [data['lat'].mean() + x * data['lat'].std() for x in [-ndesv, ndesv]]
        bounds_lon = [data['lon'].mean() + x * data['lon'].std() for x in [-ndesv, ndesv]]
        
        data = data[((data['lat']>=bounds_lat[0]) & (data['lat']<=bounds_lat[1]))
                      & ((data['lon']>=bounds_lon[0]) & (data['lon']<=bounds_lon[1]))]
    return data


def split_geolife(
    path_geolife_data: str=os.path.join(PATH_LOCAL_DATA, 'geolife_consolidated.parquet'),
    path_split_data: str=os.path.join(PATH_LOCAL_DATA, 'users'),
    user_cols: list=['time','lat','lon','user'],
    clean_weekends: bool=False, 
    clean_outliers: bool=False 
):
    if clean_weekends or clean_outliers:
        path_split_data = path_split_data + '_cl'

    if not os.path.exists(path_split_data):
        os.mkdir(path_split_data)

    geolife_data = pd.read_parquet(path_geolife_data)
    geolife_data = geolife_data.rename(columns={'hour':'time', 'lat':'lat','lng':'lon', 'user':'user'})
    
    users = sorted(geolife_data['user'].unique())
    for user in tqdm(users):
        user_data = geolife_data[geolife_data['user']==user][user_cols]

        user_data = clean_data_geopoints(user_data, clean_weekends, clean_outliers, ndesv=2)

        path_user_data = os.path.join(path_split_data, f'data_user_{user:03d}.csv')
        user_data.to_csv(path_user_data, index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cleanweekends', '-cw', type = bool, help = "Flag to delete the datapoints of the weekends" )
    parser.add_argument('--cleanoutliers','-co', type = bool, help = "Flag to delete the outlier datapoints")

    args = parser.parse_args()

    flag_clean_weekends = args.cleanweekends
    flag_clean_outliers = args.cleanoutliers

    split_geolife(
        clean_weekends=flag_clean_weekends,
        clean_outliers=flag_clean_outliers
    )