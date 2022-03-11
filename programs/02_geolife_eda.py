import os
import sys
import warnings
sys.path.append('../')

import pandas as pd

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


