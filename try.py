import json
import os
import pandas as pd

print("Hello World")

path_experiments = 'data/users_gan'
exp_folders = os.listdir(path_experiments)

l_report = []
for folder in exp_folders:
    try:
        reg_info = json.load(open(os.path.join(path_experiments, folder, 'registry_info.json'), 'r'))
        l_report.append(reg_info)
    except:
        pass

df_report = pd.DataFrame(l_report)
df_report.to_csv('report_experiments.csv', index=False)
