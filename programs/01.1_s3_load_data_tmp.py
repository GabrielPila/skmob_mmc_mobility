import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

from settings import session
from config import (PATH_S3_DATA, 
                    PATH_S3_EXPERIMENTS,
                    PATH_LOCAL_DATA, 
                    PATH_LOCAL_EXPERIMENTS)
import awswrangler as wr
import pandas as pd

def load_parquet_to_s3(filename='geolife_consolidated.parquet',
                        local_path=PATH_LOCAL_DATA, 
                        s3_path=PATH_S3_DATA):
    df = pd.read_parquet(os.path.join(local_path, filename))

    wr.s3.to_parquet(df,
        path=os.path.join(s3_path,filename), 
        boto3_session=session
    )

for file in ['geolife_consolidated.parquet','geolife_sample.parquet']:
    load_parquet_to_s3(file)