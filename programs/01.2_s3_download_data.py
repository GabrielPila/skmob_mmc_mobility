import os

from gan_utils.s3_utils import download_file_from_s3
from config import PATH_LOCAL_DATA, PATH_S3_DATA

for path_dir in ['data','experiment']:
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)


for filename in ['geolife_sample.parquet', 'geolife_consolidated.parquet']:
    download_file_from_s3(
        filename=filename, 
        local_path=PATH_LOCAL_DATA, 
        s3_path=PATH_S3_DATA, 
        print_progress=True
    )