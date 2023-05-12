import pandas as pd
import sys
import os
sys.path.insert(0, os.getcwd())
from data_extraction import  load_csv_data
import clean_data

def checkColumnNames(data_df, column_names):
    '''
    Function to check for expected number of variables 
    and write dataframe to file
    
    Args:
        data_df (pandas dataframe): dataframe
        column_names (pandas series): 1D ndarray
    Raises:
        ValueError: Missing variable
    '''
    
    if(data_df.columns.isin(column_names).sum() != len(column_names)):
        raise ValueError(f'There is a missing column in {column_names}.')
    else:
        data_df.to_csv("dataset/clean_data.csv")

file_path = "dataset/data.csv"
data_df = load_csv_data.loadCSVFile(file_path)
data_df = clean_data.cleanData(data_df)
#checkColumnNames(data_df, data_df.columns)