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

#Remove the non-feature variables
variables_to_omit = ['id', 'diagnosis']
input_data = data_df.drop(variables_to_omit, axis = 1)


def createHeatmap():
    sns.set_theme(style ='white')
    #Generate a mask for the upper triangular matrix
    mask = np.triu(input_data.corr(), k = 0)

    fig = plt.figure(figsize = (18, 18))
    ax = fig.add_subplot()

    # Generate a custom diverging palette of colours
    cmap = sns.diverging_palette(230, 20, as_cmap = True)

    sns.heatmap(data = input_data.corr(), 
                annot = True, 
                linewidths = 0.5, 
                fmt = '.1f',
                ax = ax, 
                mask = mask,
                cmap = cmap)

    plt.title('A correlation heatmap of the features', fontsize = 20)
    plt.show()

createHeatmap()