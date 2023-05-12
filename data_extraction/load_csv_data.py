import pandas as pd
import sys

file_path = "dataset/data.csv"

def loadCSVFile(file_path):
    '''
    Function to load data in csv format
    Args:
        file_path specifies location of file in computer's file system structure 
    Returns:
        data_df (pandas dataframe): dataframe containing dataset
    '''
    
    try:
        data_df = pd.read_csv(file_path)
    except OSError as e:
        print(f'Unable to open {file_path}:\n{e}',
             file = sys.stderr)
    else:
        return data_df

data_df = loadCSVFile(file_path)