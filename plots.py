# create receiver operating characteristic(ROC) curve
# i.e. how well is our model performing?
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
    # plt.show()
    # plt.save?
    plt.savefig('coderep_ROC.png')

# -----------------------    

# Confusion matrix plot
def displayConfusionMatrix():
    disp = ConfusionMatrixDisplay(confusion_matrix = cm,
                                  display_labels = model.classes_)
    
    disp.plot()
    plt.grid(visible = False)
    plt.title('Confusion matrix')
    # plt.show()
    plt.savefig('coderep_ConfMat.png')

# -----------------------    

# Logistic plot
def logistic_regression_plot(features):
    fig = plt.figure(figsize = (11, 5))
    for i, feature in enumerate(features):
        ax = fig.add_subplot(1, 3, i + 1)
        sns.regplot(data = data_df,
                    x = feature, 
                    y = Diagnosis, 
                    logistic = True, 
                    color = 'black',
                    line_kws = {'lw' : 1, 'color' : 'red'},
                    label = str(feature.replace('_', ' ').capitalize()))
        ax.set_xlabel(str(feature.replace('_', ' ').capitalize()))
        plt.ylabel('Probability')
        plt.title('Logistic regression')
        plt.legend()
    
        plt.tight_layout()
        # plt.show()
        plt.savefig('coderep_Log_plot.png')

    return None

# ---------------
# Box & whisker plot routine
def makeBoxplot(features):
    fig = plt.figure(figsize = (8, 12))
    for i, feature in enumerate(features):
        ax = fig.add_subplot(2, 2, i + 1)
        sns.boxplot(x = Diagnosis, 
                   y = feature, 
                   data = data_df, 
                   showfliers = True)
        plt.title(str(feature.replace('_', ' ').capitalize()))
        ax.set_xticklabels(xtickmarks)
        ax.set_xlabel(Diagnosis.capitalize())
        ax.set_ylabel(str(feature.replace('_', ' ').capitalize()))
        ax.grid(False)
    
    fig.tight_layout()
    plt.show()

