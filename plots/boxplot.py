import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
sys.path.insert(0, os.getcwd())

from data_extraction import load_csv_data
from data_cleaning import clean_data

file_path = "dataset/data.csv"
data_df = load_csv_data.loadCSVFile(file_path)
data_df = clean_data.cleanData(data_df)

file_path = "dataset/data.csv"
data_df = load_csv_data.loadCSVFile(file_path)
data_df = clean_data.cleanData(data_df)

worst_mean_se = ['area_worst', 'fractal_dimension_mean', 'radius_se']
xtickmarks = ['B', 'M']
# Create box and whiskers plot for texture mean by diagnosis of tumour
Diagnosis = 'diagnosis'

def makeBoxplot(features):
    fig = plt.figure(figsize = (8, 12))
    for i, feature in enumerate(features):
        ax = fig.add_subplot(2, 2, i + 1)
        sns.boxplot(x = Diagnosis, 
                   y = feature, 
                   data = data_df, 
                   showfliers = True)
        plt.title(str(feature.replace('_', ' ').capitalize()))
        ax.set_xticklabels(xtickmarks)
        ax.set_xlabel(Diagnosis.capitalize())
        ax.set_ylabel(str(feature.replace('_', ' ').capitalize()))
        ax.grid(False)
    
    fig.tight_layout()
    plt.show()

makeBoxplot(worst_mean_se)