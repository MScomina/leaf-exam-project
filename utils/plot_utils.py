import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_confusion_matrix(confusion_matrix, plot_name : str | None = None, n_classes=30, figsize = (10,7), fontsize=10):

    df_cm = pd.DataFrame(
        confusion_matrix, index=list(range(n_classes)), columns=list(range(n_classes)), 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.title(f'Confusion matrix ({plot_name})')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if plot_name != None:
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{plot_name.lower().replace(" ", "_")}.png')
    else:
        plt.show()
    plt.close(fig)