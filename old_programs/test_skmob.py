import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import skmob
from tqdm import tqdm
from skmob.preprocessing import (filtering, 
                                 detection, 
                                 compression, 
                                 clustering)

geo = pd.read_csv('../data/geo82.csv.zip')
geo.columns = ['user', 'hour', 'lat', 'lng']

geo['user'] = geo['user'].map(int)
 
print(geo.shape)
print(geo.head())

tdf = skmob.TrajDataFrame(geo, 
                            datetime='hour', 
                            user_id='user')

print(tdf.shape)
print(tdf.head())


tdf_f = filtering.filter(tdf,
                        max_speed_kmh=1)

print(tdf_f.shape)
print(tdf_f.head())
