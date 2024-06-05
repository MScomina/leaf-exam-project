import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_confusion_matrix(confusion_matrix, plot_name : str | None = None, n_classes=30, figsize = (10,7), fontsize=10, compact=False):

    df_cm = pd.DataFrame(
        confusion_matrix, index=list(range(n_classes)), columns=list(range(n_classes)), 
    )
    if compact and figsize == (10,7):
        figsize = (7,7)
    fig = plt.figure(figsize=figsize)
    try:
        if compact:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, cmap="Greys_r")
        else:
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
        plt.savefig(f'plots/cm_{plot_name.lower().replace(" ", "_")}.png')
    else:
        plt.show()
    plt.close(fig)

def plot_n_confusion_matrices(confusion_matrices : tuple, plot_names : list[str], n_classes=30, figsize = (10,7), fontsize=10, compact=False):
    if len(confusion_matrices) != len(plot_names):
        raise ValueError("Number of confusion matrices and plot names must be equal.")
    if compact and figsize == (10,7):
        figsize = (7, 7)
    figsize = (figsize[0] * len(confusion_matrices), figsize[1])
    fig, axs = plt.subplots(1, len(confusion_matrices), figsize=figsize)
    for i, (confusion_matrix, plot_name) in enumerate(zip(confusion_matrices, plot_names)):
        df_cm = pd.DataFrame(
            confusion_matrix, index=list(range(n_classes)), columns=list(range(n_classes)), 
        )
        if compact:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, cmap="Greys_r", ax=axs[i])
        else:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", ax=axs[i])
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        axs[i].set_title(f'Confusion matrix ({plot_name})')
        if i == 0:
            axs[i].set_ylabel('True label')
        else:
            axs[i].set_yticklabels([])
        axs[i].set_xlabel('Predicted label')
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/confusion_matrices.png')
    plt.close(fig)

def plot_roc_curve(fpr, tpr, plot_name : str = None, figsize = (10,7), fontsize=10, compact=False):
    if compact and figsize == (10,7):
        figsize = (7,7)
    plt.figure(figsize=figsize)
    plt.plot(fpr["micro"], tpr["micro"], color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.title('Receiver Operating Characteristic' + (f' - {plot_name}' if plot_name else ''), fontsize=fontsize)
    plt.legend(loc="lower right", fontsize=fontsize)
    if plot_name != None:
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/roc_{plot_name.lower().replace(" ", "_")}.png')
    else:
        plt.show()
    plt.close()

def plot_n_roc_curves(fpr_list, tpr_list, plot_names : list[str], figsize = (10,7), fontsize=10, compact=False):
    if len(fpr_list) != len(tpr_list) or len(fpr_list) != len(plot_names):
        raise ValueError("Number of FPR, TPR lists and plot names must be equal.")
    if compact and figsize == (10,7):
        figsize = (7, 7)
    fig = plt.figure(figsize=figsize)
    for fpr, tpr, plot_name in zip(fpr_list, tpr_list, plot_names):
        plt.plot(fpr["micro"], tpr["micro"], lw=2, label=plot_name)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.title('Receiver Operating Characteristic', fontsize=fontsize)
    plt.legend(loc="lower right", fontsize=fontsize)
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/roc_curves.png')
    plt.close(fig)