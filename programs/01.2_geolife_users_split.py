import pandas as pd
import numpy as np
import os
from config import PATH_LOCAL_DATA
from tqdm import tqdm


def split_geolife(
    path_geolife_data: str=os.path.join(PATH_LOCAL_DATA, 'geolife_consolidated.parquet'),
    path_split_data: str=os.path.join(PATH_LOCAL_DATA, 'users'),
    user_cols: list=['time','lat','lon','user']
):
    if not os.path.exists(path_split_data):
        os.mkdir(path_split_data)

    geolife_data = pd.read_parquet(path_geolife_data)
    geolife_data = geolife_data.rename(columns={'hour':'time', 'lat':'lat','lng':'lon', 'user':'user'})
    
    users = sorted(geolife_data['user'].unique())
    for user in tqdm(users):
        user_data = geolife_data[geolife_data['user']==user][user_cols]

        path_user_data = os.path.join(path_split_data, f'data_user_{user:03d}.csv')
        user_data.to_csv(path_user_data, index=False)

if __name__ == '__main__':
#    pass
    split_geolife()