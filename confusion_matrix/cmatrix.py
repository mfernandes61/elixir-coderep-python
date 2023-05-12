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
cm = confusion_matrix(logit_model.y_test, model.predict(logit_model.X_test))

def displayConfusionMatrix():
    disp = ConfusionMatrixDisplay(confusion_matrix = cm,
                                  display_labels = model.classes_)
    
    disp.plot()
    plt.grid(visible = False)
    plt.title('Confusion matrix')
    plt.show()


displayConfusionMatrix()