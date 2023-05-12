import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # Hypthesis testing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler #for robust feature scaling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, roc_curve, auc

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

#Remove the non-feature variables
variables_to_omit = ['id', 'diagnosis']
input_data = data_df.drop(variables_to_omit, axis = 1)

X = input_data
Y = data_df.diagnosis

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    Y, 
                                                    test_size = 0.20, 
                                                    stratify = Y, 
                                                    random_state = 1234)
#Robust feature scaling
rs_object = RobustScaler()
X_train = rs_object.fit_transform(X_train)
X_test = rs_object.transform(X_test)

# Define a function which trains a logistic model
def createModel(X_train, y_train):
    
    
    LogitModel = LogisticRegression(solver = 'lbfgs', 
                             max_iter = 100, 
                             random_state = 1234)
    
    LogitModel.fit(X_train, y_train)  
    
    #Display model accuracy on the training data.
    print(f'Accuracy for the training sample: {LogitModel.score(X_train, y_train):.2f}')
    return LogitModel

#Obtain the training results
model = createModel(X_train, y_train)