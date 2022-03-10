import os

from config import PATH_LOCAL_DATA, PATH_S3_DATA
from gan_utils.s3_utils import upload_file_to_s3 

for path_dir in ['data','experiment']:
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)


for filename in ['geolife_sample.parquet', 'geolife_consolidated.parquet']:
    upload_file_to_s3(
        filename=filename, 
        local_path=PATH_LOCAL_DATA, 
        s3_path=PATH_S3_DATA, 
        print_progress=True
    )    

