import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import awswrangler as wr
import boto3
import pandas as pd

from config import (PATH_S3_DATA, 
                    PATH_S3_EXPERIMENTS,
                    PATH_LOCAL_DATA,
                    PATH_LOCAL_EXPERIMENTS)
from settings import session


def download_parquet_from_s3(filename:str='geolife_consolidated.parquet',
                            local_path:str=PATH_LOCAL_DATA, 
                            s3_path:str=PATH_S3_DATA):
    '''Load a parquet object stored in S3 as a dataframe and then saves it in Local'''
    path_file_s3 = os.path.join(s3_path, filename)
    path_file_local = os.path.join(local_path, filename)
    df = wr.s3.read_parquet(
        path=path_file_s3, 
        boto3_session=session
    )
    df.to_parquet(path_file_local)


def upload_parquet_to_s3(filename:str='geolife_consolidated.parquet',
                        local_path:str=PATH_LOCAL_DATA, 
                        s3_path:str=PATH_S3_DATA):
    '''Read a parquet object stored local and uploads it to S3 using awswrangler'''
    df = pd.read_parquet(os.path.join(local_path, filename))

    wr.s3.to_parquet(df,
        path=os.path.join(s3_path,filename), 
        boto3_session=session
    )


def upload_file_to_s3(filename:str='geolife_consolidated.parquet',
                      local_path:str=PATH_LOCAL_DATA, 
                      s3_path:str=PATH_S3_DATA,
                      print_progress:bool=False):
    '''Uploads a file from a Local path to an S3 URI
    
    Parameters
    ----------
        filename (str): name of the file to upload (only name with extension)
        local_path (str): path of the local directory where the file is located
        s3_path (str): URI of the S3 directory where the file will be uploaded
    '''
    s3 = boto3.resource('s3')
    path_file_s3 = os.path.join(s3_path, filename)
    path_file_local = os.path.join(local_path, filename)
    
    bucket_s3, key_s3 = path_file_s3[5:].split('/', 1) # split into bucket and key
    
    s3.Bucket(bucket_s3).upload_file(path_file_local, key_s3)
    if print_progress:
        print(f'INFO: File uploaded to {path_file_s3}')    


def download_file_from_s3(filename:str='geolife_consolidated.parquet',
                          local_path:str=PATH_LOCAL_DATA, 
                          s3_path:str=PATH_S3_DATA,
                          print_progress:bool=False):
    '''Downloads a file from S3 to a Local path
    
    Parameters
    ----------
        filename (str): name of the file to download (only name with extension)
        local_path (str): path of the local directory where the file will be downloaded
        s3_path (str): URI of the S3 directory where the file is located
    '''
    s3 = boto3.resource('s3')
    path_file_s3 = os.path.join(s3_path, filename)
    path_file_local = os.path.join(local_path, filename)
        
    bucket_s3, key_s3 = path_file_s3[5:].split('/', 1) # split into bucket and key

    s3.Bucket(bucket_s3).download_file(key_s3, path_file_local)
    if print_progress:
        print(f'INFO: File download from {path_file_s3}')

