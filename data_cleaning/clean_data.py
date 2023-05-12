import pandas as pd

import sys
import os
sys.path.insert(0, os.getcwd())
from data_extraction import  load_csv_data

def cleanData(data_df):
    '''
    Function to clean data
    Args:
        data_df (pandas dataframe): dataframe
    Returns:
        data_df (pandas dataframe): dataframe of cleaned data with no missing values
    '''

    assert len(data_df.index) != 0, 'The input dataframe is empty'
    assert data_df.isnull().sum().sum() != 0, 'There are some missing values'

    data_df.diagnosis = data_df.diagnosis.apply(lambda x : 1 if x == 'M' else 0)
    data_df = data_df.dropna(axis = 1)

    assert data_df.notnull().all().all(), 'There are no missing values'
    
    return data_df

file_path = "dataset/data.csv"
data_df = load_csv_data.loadCSVFile(file_path)
data_df = cleanData(data_df)