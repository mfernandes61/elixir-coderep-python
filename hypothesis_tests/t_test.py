import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # Hypthesis testing

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
Diagnosis = 'diagnosis'

class Hypothesis_T_Test(object):
    def __init__(self, feature, ind_variable = Diagnosis):
        self.feature = feature
        self.ind_variable = ind_variable
        
    def computeTandPValues(self):
        hypothesis_test_data = pd.DataFrame(data = data_df[[self.feature, self.ind_variable]])
        hypothesis_test_data = hypothesis_test_data.set_index(self.ind_variable)
        self.variable_name = lambda : data_df[self.feature].name.replace('_', ' ').capitalize()
        self.t_value, self.p_value = stats.ttest_ind(hypothesis_test_data.loc[0], hypothesis_test_data.loc[1])
        print(f'Variable name: {self.variable_name()}: t-value: {self.t_value}, p-value: {self.p_value}')
        
        return self.t_value, self.p_value

for feature in worst_mean_se:
    HTT = Hypothesis_T_Test(feature)
    HTT.computeTandPValues()