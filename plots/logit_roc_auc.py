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
from models import logit_model

file_path = "dataset/data.csv"
data_df = load_csv_data.loadCSVFile(file_path)
data_df = clean_data.cleanData(data_df)


model = logit_model.createModel(logit_model.X_train, logit_model.y_train)


# Compute predicted probabilities and keep results only for positive outcome 
y_pred_prob = model.predict_proba(logit_model.X_test)[:,1]
# Generate ROC curve values and capture only fpr, and tpr, but not thresholds
fpr, tpr, _ = roc_curve(logit_model.y_test, y_pred_prob)

print(f'The AUC score for the logistic regression model is: {auc(fpr, tpr):.4f}')

def createROC():
    fig = plt.figure()
    ax = fig.add_subplot()

    plt.plot([0, 1], [0, 1], 'k-.', label = 'Random prediction')
    plt.plot(fpr, tpr, label = 'Logistic regression model: AUC = %0.4f' % auc(fpr, tpr))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Logistic Regression')

    ax.grid(False)
    plt.legend()
    plt.show()

createROC()