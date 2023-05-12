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

Malignant = data_df[data_df['diagnosis'] == 1]
Benign = data_df[data_df['diagnosis'] == 0]

worst_mean_se = ['area_worst', 'fractal_dimension_mean', 'radius_se']

def makeHistogram(features):
    '''
    Function to create a histogram of 3 feature variables
    
    Args: 
        features list of 3 feature variables
    '''
    
    for feature in features:
        if not type(feature) is str:
            raise TypeError('Only strings are permitted')
            
    fig = plt.figure(figsize = (10, 8))
    for i, feature in enumerate(features):
        ax = fig.add_subplot(1, 3, i + 1)  
        sns.histplot(Malignant[feature], 
                   bins = bins, 
                   color = 'red', 
                   label = 'Malignant',
                   kde = True)
        sns.histplot(Benign[feature], 
                   bins = bins, 
                   color = 'green', 
                   label = 'Benign',
                   kde = True)
        plt.title(str(' Distribution of  ') + str(feature.replace('_', ' ').capitalize()))
        plt.xlabel(str(feature.replace('_', ' ').capitalize()))
        plt.ylabel('Density function')
        plt.legend(loc = 'upper right')
        ax.grid(False)
    
    plt.tight_layout()
    plt.show()

bins = 'fd' #Freedman and Diaconis 


makeHistogram(worst_mean_se)