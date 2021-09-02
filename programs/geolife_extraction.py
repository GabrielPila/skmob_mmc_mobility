# This program will run once having the zip file of Geolife Trajectories
# The zip file should be in the following path: '../data/Geolife Trajectories 1.3.zip'
# Then, unzip the file to have the Geolife Trajectories folder in this directory

import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
import os


warnings.filterwarnings('ignore')

id_trajs = os.listdir('../data/Geolife Trajectories 1.3/Data')
id_trajs = sorted([x for x in id_trajs if '.' not in x])

re_start = 0
uid_start = id_trajs[0]

# Data Extraction - GEOLIFE
data = pd.DataFrame()
l = 0
for user_id in tqdm(id_trajs):
    
    if re_start == 1: 
        uid_start = user_id
        re_start = 0
        
    path_geo = f'../data/Geolife Trajectories 1.3/Data/{user_id}/Trajectory/'
    list_files = os.listdir(f'{path_geo}')
    list_files = sorted([x for x in list_files if '.plt' in x])
    
    data_user = pd.DataFrame()
    for file in list_files:
        path_file = f'{path_geo}{file}'
        di = pd.read_csv(f'{path_file}', skiprows=5).reset_index()
        
        l += di.shape[0]
        
        di['file'] = file
        di['user_id'] = user_id
        data_user = data_user.append(di)
        
    data = data.append(data_user).reset_index(drop=True)
    
    if (int(user_id) % 10 == 9) or (user_id==id_trajs[-1]):
        re_start = 1
        data.columns = ['lat','lng','dummy', 'alt', 'date_days',
                        'date', 'time','file','user_id']
        
        data.to_csv(f'../data/geo_{uid_start}_{user_id}.csv.zip', 
                    compression='zip', index=False)
        print(f'{uid_start}_{user_id}')
        
        data = pd.DataFrame()