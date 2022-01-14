import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from tqdm import tqdm
import sys
sys.path.append('../')
import os


def plot_histogram(serie, 
                   title='Distribución de eventos según drift en segundos (log10)',
                   ylabel='Numero de Eventos', 
                   xlabel='Time Drift (log10 scale)',
                   legend=['events_geolife'],
                   loc_legend='upper right',
                   save_path='../img/',
                   filename='image.jpg',
                   font_title=15,
                   font_xlabel=12,
                   font_ylabel=12,
                   bins=25, 
                   color='r', 
                   edgecolor='k',
                   figsize=[10,6], 
                   alpha=0.25):    
    img_path = os.path.join(save_path, filename)

    plt.figure(figsize=figsize)
    plt.hist(x=[serie], bins=bins, color=color, edgecolor=edgecolor)
    plt.grid(axis='y', alpha=alpha)
    plt.title(title, fontsize=font_title);
    plt.ylabel(ylabel, fontsize=font_ylabel)
    plt.xlabel(xlabel, fontsize=font_xlabel)
    plt.legend(legend, loc=loc_legend)
    plt.savefig(img_path)
    plt.plot();




def plot_event_distribution_per_time_drift(data):
    
    serie_seconds_events = data['seconds_diff'].value_counts()
    serie_seconds_events_log = np.log10(serie_seconds_events)

    # PLOT
    plot_histogram(serie=serie_seconds_events_log, 
                   title='Distribución de usuarios según Nro de Eventos (log10)',
                   ylabel='Numero de Usuarios', 
                   xlabel='Numero de Eventos (log10 scale)',
                   legend=['users_geolife'],
                   loc_legend='upper right',
                   save_path='../img/',
                   filename='user_distribution_per_events.jpg')