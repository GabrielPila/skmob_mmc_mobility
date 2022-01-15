import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import skmob
from tqdm import tqdm
import sys
sys.path.append('../')
from src.geo_utils import (get_clusters_from_tdf,
                            assign_tdf_points_to_clusters,
                            get_mmc_transitions,
                            get_stationary_vector)
import os
from src.geo_plots import (plot_user_distribution_by_events,
                            plot_event_distribution_by_time_drift,
                            plot_event_distribution_by_space_drift)

warnings.filterwarnings('ignore')


# Loading Data
data = pd.read_parquet('../data/geolife_consolidated.parquet')
print("INFO: The data has been loaded")

plot_user_distribution_by_events(data)
plot_event_distribution_by_time_drift(data)
plot_event_distribution_by_space_drift(data)


