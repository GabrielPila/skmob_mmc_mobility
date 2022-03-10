import json
import os

import pandas as pd

from config import PATH_LOCAL_DATA, PATH_S3_EXPERIMENTS
from config import dir_user_output
from gan_utils.s3_utils import upload_file_to_s3

print("Hello World")

path_experiments = os.path.join(PATH_LOCAL_DATA, dir_user_output)
exp_folders = os.listdir(path_experiments)

l_report = []
for folder in exp_folders:
    try:
        reg_info = json.load(open(os.path.join(path_experiments, folder, 'registry_info.json'), 'r'))
        l_report.append(reg_info)
    except:
        pass

df_report = pd.DataFrame(l_report)

report_filename = 'report_experiments.csv'
path_report_dir = os.path.join(path_experiments, report_filename)
df_report.to_csv(path_report_dir, index=False)

# Upload data to S3
upload_file_to_s3(
    filename=report_filename, 
    local_path=path_experiments, 
    s3_path=PATH_S3_EXPERIMENTS, 
    print_progress=True
)
