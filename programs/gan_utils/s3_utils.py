import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import pandas as pd
import awswrangler as wr
import boto3

from settings import session
from config import (PATH_S3_DATA, 
                    PATH_S3_EXPERIMENTS,
                    PATH_LOCAL_DATA,
                    PATH_LOCAL_EXPERIMENTS)


def download_parquet_from_s3(filename='geolife_consolidated.parquet',
                            local_path=PATH_LOCAL_DATA, 
                            s3_path=PATH_S3_DATA):

    path_file_s3 = os.path.join(s3_path, filename)
    path_file_local = os.path.join(local_path, filename)
    df = wr.s3.read_parquet(
        path=path_file_s3, 
        boto3_session=session
    )
    df.to_parquet(path_file_local)


def upload_parquet_to_s3(filename='geolife_consolidated.parquet',
                        local_path=PATH_LOCAL_DATA, 
                        s3_path=PATH_S3_DATA):
    df = pd.read_parquet(os.path.join(local_path, filename))

    wr.s3.to_parquet(df,
        path=os.path.join(s3_path,filename), 
        boto3_session=session
    )


def upload_file_to_s3(filename='geolife_consolidated.parquet',
                      local_path=PATH_LOCAL_DATA, 
                      s3_path=PATH_S3_DATA,
                      print_progress=False):
    s3 = boto3.resource('s3')
    path_file_s3 = os.path.join(s3_path, filename)
    path_file_local = os.path.join(local_path, filename)
    
    bucket_s3, key_s3 = path_file_s3[5:].split('/', 1) # split into bucket and key
    
    s3.Bucket(bucket_s3).upload_file(path_file_local, key_s3)
    if print_progress:
        print(f'INFO: File uploaded to {path_file_s3}')    


def download_file_from_s3(filename='geolife_consolidated.parquet',
                          local_path=PATH_LOCAL_DATA, 
                          s3_path=PATH_S3_DATA,
                          print_progress=False):
    s3 = boto3.resource('s3')
    path_file_s3 = os.path.join(s3_path, filename)
    path_file_local = os.path.join(local_path, filename)
        
    bucket_s3, key_s3 = path_file_s3[5:].split('/', 1) # split into bucket and key

    s3.Bucket(bucket_s3).download_file(key_s3, path_file_local)
    if print_progress:
        print(f'INFO: File download from {path_file_s3}')

