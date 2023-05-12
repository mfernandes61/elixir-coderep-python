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

Diagnosis = 'diagnosis'
worst_mean_se = ['area_worst', 'fractal_dimension_mean', 'radius_se']

def logistic_regression_plot(features):
    fig = plt.figure(figsize = (11, 5))
    for i, feature in enumerate(features):
        ax = fig.add_subplot(1, 3, i + 1)
        sns.regplot(data = data_df,
                    x = feature, 
                    y = Diagnosis, 
                    logistic = True, 
                    color = 'black',
                    line_kws = {'lw' : 1, 'color' : 'red'},
                    label = str(feature.replace('_', ' ').capitalize()))
        ax.set_xlabel(str(feature.replace('_', ' ').capitalize()))
        plt.ylabel('Probability')
        plt.title('Logistic regression')
        plt.legend()
    
        plt.tight_layout()
        plt.show
    
    return None

logistic_regression_plot(worst_mean_se)