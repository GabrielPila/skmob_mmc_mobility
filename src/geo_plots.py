import sys
import os
import warnings

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

sys.path.append('../')


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




def plot_user_distribution_by_events(data):
    '''Plot User Distribution
    Plots the distribution of users according to the number of events they have in Geolife'''
    
    serie_user_events = data['user'].value_counts()
    serie_user_events_log = np.log10(serie_user_events)
    
    plot_histogram(serie=serie_user_events_log, 
               title='Distribución de usuarios según Nro de Eventos (log10)',
               ylabel='Numero de Usuarios', 
               xlabel='Numero de Eventos (log10 scale)',
               legend=['users_geolife'],
               loc_legend='upper right',
               save_path='../img/',
               filename='user_distribution_per_events.jpg')


def plot_event_distribution_by_time_drift(data):
    '''Plot User Distribution
    Plots the distribution of users according to the number of events they have in Geolife'''

    # Distribución de número de eventos
    serie_seconds_events = data['seconds_diff'].value_counts()
    serie_seconds_events_log = np.log10(serie_seconds_events)

    plot_histogram(serie=serie_seconds_events_log, 
               title='Distribución de eventos según drift en segundos (log10)',
               ylabel='Numero de Eventos', 
               xlabel='Delta de Segundos (log10 scale)',
               legend=['events_geolife'],
               loc_legend='upper right',
               save_path='../img/',
               filename='event_distribution_by_time_drift.jpg')            


def plot_event_distribution_by_space_drift(data):
    '''Plot User Distribution
    Plots the distribution of users according to the number of events they have in Geolife'''

    # Distribución de número de eventos
    serie_distance_events = data['distance_to_last_km'].value_counts()
    serie_distance_events_log = np.log10(serie_distance_events)
    
    plot_histogram(serie=serie_distance_events_log, 
               title='Distribución de eventos según drift de distancia (log10)',
               ylabel='Numero de Eventos', 
               xlabel='Delta de KM (log10 scale)',
               legend=['events_geolife'],
               loc_legend='upper right',
               save_path='../img/',
               filename='event_distribution_by_space_drift.jpg')               