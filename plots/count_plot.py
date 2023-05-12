import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def createCountplot(diagnosis):
    
    '''
    Function to display a count plot of the tumours
    Args: 
        diagnosis (pandas series): 1D ndarray
    '''
    
    xtickmarks = ['B', 'M']
    
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_subplot()

    sns.set_theme(style = 'whitegrid')

    sns.countplot(data = data_df,
                  x = data_df.diagnosis,
                  label = 'Count',
                  lw = 4,
                  ec = 'black').set(title = 'A count of benign and malignant tumours',
                                    xlabel = 'Diagnosis',
                                    ylabel = 'Count')

    ax.set_xticklabels(xtickmarks)
    plt.show()


createCountplot(data_df.diagnosis)