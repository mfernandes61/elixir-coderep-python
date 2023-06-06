#Packages required 

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler #for robust feature scaling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, roc_curve, auc
from scipy import stats # Hypthesis testing 
import seaborn as sns
# add in special sauce to use our python files containing useful functions
import plots as pe
import models as mods
import html_gen as hg

# Read and characterise data
data_df = pd.read_csv("datasets/data.csv")

data_df.head().T

hg.start_html() # start creating web page

data_df.diagnosis = data_df.diagnosis.apply(lambda x: 1 if x == 'M' else 0)

data_df.shape
data_df.columns
data_df = data_df.dropna(axis = 1)

B, M = data_df.diagnosis.value_counts()
xtickmarks = ['B', 'M']

outp = 'Number of Malignant tumours: ' + M +'\n'
outp = outp + 'Number of Benign tumours   : ' + B + '\n'

hg.html_para(outp)
pe.createCountplot(data_df, xtickmarks)

hg.html_image("Countplot", "coderep_countplot.png")

variables_to_omit = ['id', 'diagnosis']
input_data = data_df.drop(variables_to_omit, axis = 1)
r, c = input_data.shape

outp = 'Sample size                    : ' + r + '\n'
outp = outp + 'Number of independent variables: ' + c + '\n'
hg.html_para(outp)

# Histograms
# 
Malignant = data_df[data_df['diagnosis'] == 1]
Benign = data_df[data_df['diagnosis'] == 0]
worst_mean_se = ['area_worst', 'fractal_dimension_mean', 'radius_se']

bins = 'fd'

pe.makeHistogram(worst_mean_se, Malignant,Benign,  bins)
hg.html_image("Histogram", "coderep_histogram.png")

# **Heatmaps** provide an informative way to depict two-dimensional data of the kind we have before us. A *heatmap* is an image in which the colour of each pixel is determined by the corresponding value in the array of data. 

pe.createHeatmap(input_data)
hg.html_image("Heatmap", "coderep_heatmap.png")

# More plots

# Create box and whiskers plot for texture mean by diagnosis of tumour
Diagnosis = 'diagnosis'

pe.makeBoxplot(worst_mean_se, data_df, Diagnosis, xtickmarks)
hg.html_image("Boxplot", "coderep_boxplot.png")

# Logistic regression plots

# Some more box and whiskers plots

pe.logistic_regression_plot(worst_mean_se, data_df, Diagnosis)
hg.html_image("Logistic regression plot", "coderep_log_plot.png")

# Hypothesis testing using Student's t-test.

# Make a new dataframe with only the desired feature for t test  
hypothesis_test_data = pd.DataFrame(data = data_df[['area_worst', 'diagnosis']])
hypothesis_test_data = hypothesis_test_data.set_index(Diagnosis)
t, p = stats.ttest_ind(hypothesis_test_data.loc[0], hypothesis_test_data.loc[1])

outp = '\nThe t-value: ' + t + '\n'
outp = outp + 'The p-value: ' + p + '\n'
hg.html_para(outp)
      
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


# Some correlation 

# Create a correlation matrix
corr_matrix = input_data.corr().abs()
# Select upper triangle of correlation matrix
upper_triangular = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
# Find index of feature columns with correlation greater than 0.95
correlation_threshold = 0.95
features_to_omit = [column for column in upper_triangular.columns if any(upper_triangular[column] > correlation_threshold)]
# Remove features to omit 
correlation_data = input_data.drop(input_data[features_to_omit], axis = 1)
correlation_data.columns
outp = correlation_data.columns +  '\n'
hg.html_para(outp)

# Machine learning

X = input_data
Y = data_df.diagnosis

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    Y, 
                                                    test_size = 0.20, 
                                                    stratify = Y, 
                                                    random_state = 1234)

# Robust feature scaling
rs_object = RobustScaler()
X_train = rs_object.fit_transform(X_train)
X_test = rs_object.transform(X_test)

#Obtain the training results
model = mods.createModel(X_train, y_train)

# Create Confusion matrix

cm = confusion_matrix(y_test, model.predict(X_test))
 
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
 
outp = cm + '\n'
outp = outp + 'Accuracy on the test data: ' + (TP + TN) / (TP + TN + FN + FP) + '\n\n'
hg.html_para(outp)

# modify call to pass model.classes as parameter
pe.displayConfusionMatrix(cm, model)
hg.html_image("Confusion Matrix", "coderep_ConfMat.png")

# Classification report is used in machine learning to compute accuracy of a classification model from the values of the confusion matrix. In the classification report, precision is a measure of positive predictions.

outp = 'Logistic regression model \n'
#Check precision, recall, f1-score
outp = outp + 'Classification report \n'
outp = outp + classification_report(y_test, model.predict(X_test))
#Another way to get the models accuracy on the test data
outp = outp + '\nAccuracy score ' + accuracy_score(y_test, model.predict(X_test)) + '\n\n'
hg.html_para(outp)

# ROC Curve

# Compute predicted probabilities and keep results only for positive outcome 
y_pred_prob = model.predict_proba(X_test)[:,1]
# Generate ROC curve values and capture only fpr, and tpr, but not thresholds
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

outp = 'The AUC score for the logistic regression model is: ' + auc(fpr, tpr) + '\n'
hg.html_para(outp)

# modify called function to have params of true and false positive rate
pe.createROC(fpr,tpr)
hg.html_image("ROC curve", "coderep_ROC.png")

# Finish up the web page
hg.end_html()
