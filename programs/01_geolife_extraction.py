###################### STEPS #########################
# 1. Download data from Microsoft
# 2. Unzip Geolife Data
# 3. Consolidate the datasets from the different files
# 4. Save the Extracted data
# 5. Estimate the distances between contiguous points
# 6. Save the consolidated data
######################################################

import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
import os
import skmob
import gc

warnings.filterwarnings('ignore')

PATH_GEO_DATA = './Geolife Trajectories 1.3/Data'
PATH_DATA = './data'

#################### FUNCTIONS #######################

def get_distance(row):
    '''Returns the distance between two continuous points'''
    lat, lng, lat_last, lng_last = (row['lat'], row['lng'], 
                                    row['lat_last'], row['lng_last'])
    return skmob.utils.utils.distance((lat, lng), (lat_last, lng_last))

def main():
    ################### PULLING DATA ######################

    # Get Information from Microsoft
    !wget https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip

    # Unzip Information
    !unzip -q "Geolife Trajectories 1.3.zip"



    #################### DATA CONSOLIDATION ######################

    # List the user trajectories
    id_trajs = os.listdir(PATH_GEO_DATA)
    id_trajs = sorted([x for x in id_trajs if '.' not in x])


    ### Data Extraction - GEOLIFE
    # Thw following code points to the unzipped folder of GEOLIFE

    re_start = 0
    uid_start = id_trajs[0]
    data = pd.DataFrame()
    l = 0
    for user_id in tqdm(id_trajs):
        
        if re_start == 1: 
            uid_start = user_id
            re_start = 0
            
        path_geo_id = f'{PATH_GEO_DATA}/{user_id}/Trajectory/'

        list_files = os.listdir(f'{path_geo_id}')
        list_files = sorted([x for x in list_files if '.plt' in x])
        
        data_user = pd.DataFrame()
        for file in list_files:
            path_file = f'{path_geo_id}{file}'
            di = pd.read_csv(f'{path_file}', skiprows=5).reset_index()
            
            l += di.shape[0]
            
            di['file'] = file
            di['user_id'] = user_id
            data_user = data_user.append(di)
            
        data = data.append(data_user).reset_index(drop=True)

    # Renaming columns
    data.columns = ['lat','lng','dummy', 'alt', 'date_days',
                            'date', 'time','file','user_id']

    # SAVING DATA EXTRACTED
    gc.collect()
    data.to_parquet('./data/geolife_extracted.parquet', index=False)




    #################### DATA PREPROCESSING ######################

    geo_columns = ['user', 'hour', 'lat', 'lng']

    data['hour'] = data['date'] + ' ' + data['time']
    data['hour'] = pd.to_datetime(data['hour'])
    data['user'] = data['user_id'].map(int)

    data = data[geo_columns]
    data = data.drop_duplicates()
    data = data.groupby(['user','hour'], as_index=False).nth(0)
    data = data.reset_index(drop=True)


    # Estimating distance and time differences
    data[['lat_last','lng_last']] = data.groupby('user').shift(1)[['lat','lng']]
    data['seconds_diff'] = data.groupby('user')['hour'].diff(1).dt.seconds

    data['distance_to_last_km'] = data.apply(get_distance, axis=1) 
    data['speed_mps'] = data['distance_to_last_km'] / data['seconds_diff'] * 1000

    del data['lat_last'], data['lng_last']

    # SAVING DATA CONSOLIDATED
    gc.collect()
    data.to_parquet('../data/geolife_consolidated.parquet', index=False)

    # The datasets have to be loaded to GDrive via the GUI or using Drive Mount
    # Later the datasets must be prepared to be downloaded with a "gdown" command

    # Sample: !gdown https://drive.google.com/uc?id=1h9RohsM_Z9w-Ny_WHwZt856tljl5I39J

if __name__ == '__main__':
    main()