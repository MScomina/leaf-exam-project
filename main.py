#
#   Code for the "Introduction to Machine Learning and Evolutonary Robotics" project.
#
from utils.data_utils import load_data, split_data
from utils.plot_utils import plot_confusion_matrix, plot_n_confusion_matrices, plot_roc_curve, plot_n_roc_curves
from utils.training_utils import grid_search, cross_validate, train_model, test_model

import numpy as np
import random
import time

RANDOM_FOREST_PARAMS = {
    'n_estimators': [10, 25, 50, 100, 200, 500, 1000],
    'max_depth': [1, 3, 5, 10, 15, 25, 50]
}

SVC_PARAMS = {
    'C': [0.1, 0.5, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'gamma': [1.0, 'scale', 'auto']
}

NB_PARAMS = {}

CV_FOLDS = 5
TEST_RATIO = 0.2
SEED = 314
N_TESTINGS = 20

def training_script(verbose=False):

    X, y = load_data("dataset/leaf.csv")
    x_train, x_test, y_train, y_test = split_data(X, y, test_size=TEST_RATIO, random_state=SEED)

    timer = time.time()

    cm_list = []
    tpr_list = []
    fpr_list = []
    
    for model_name, model_params in zip(["RandomForest", "SVC", "NaiveBayes"], [RANDOM_FOREST_PARAMS, SVC_PARAMS, NB_PARAMS]):
        
        print(f"Training {model_name}")
        if model_name != "NaiveBayes":
            timer = time.time()
            _, best_params, _ = grid_search(model_name, model_params, x_train, y_train, cv_folds=CV_FOLDS)
            print(f"Best {model_name} Params: {best_params}")
            print(f"Grid Search Time ({model_name}): {time.time() - timer}")
        else:
            best_params = {}
            best_nb = cross_validate(model_name, x_train, y_train, cv_folds=CV_FOLDS)
            if verbose:
                print(f"Average {model_name} Accuracy: {best_nb[0]}")
        
        avg_accuracy = 0
        avg_auc = 0
        avg_train_time = 0
        avg_test_time = 0

        for k in range(N_TESTINGS):
            timer = time.time()
            model = train_model(model_name, best_params, x_train, y_train, seed=SEED+k)
            train_time = time.time() - timer
            if verbose:
                print(f"Training Time ({model_name}): {train_time}")
            timer = time.time()
            accuracy, cm, report, auc, fpr, tpr = test_model(model, x_test, y_test)
            test_time = time.time() - timer
            if verbose:
                print(f"Testing Time ({model_name}): {test_time}")
                print(f"{model_name} weighted Accuracy: {accuracy}")
                print(f"{model_name} AUC: {auc}")
                #print(f"{model_name} Classification Report:\n {report}")
            if k == 0:
                plot_confusion_matrix(cm, n_classes=30, plot_name=model_name, compact=True)
                plot_roc_curve(fpr, tpr, plot_name=model_name, compact=True)
                cm_list.append(cm)
                fpr_list.append(fpr)
                tpr_list.append(tpr)
            avg_accuracy += accuracy
            avg_auc += auc
            avg_train_time += train_time
            avg_test_time += test_time
        avg_accuracy /= N_TESTINGS
        avg_auc /= N_TESTINGS
        avg_train_time /= N_TESTINGS
        avg_test_time /= N_TESTINGS
        print(f"Average {model_name} Accuracy: {avg_accuracy}")
        print(f"Average {model_name} AUC: {avg_auc}")
        print(f"Average {model_name} Training Time: {avg_train_time}")
        print(f"Average {model_name} Testing Time: {avg_test_time}")


    plot_n_confusion_matrices(cm_list, ["RandomForest", "SVC", "NaiveBayes"], n_classes=30, compact=True)
    plot_n_roc_curves(fpr_list, tpr_list, ["RandomForest", "SVC", "NaiveBayes"], compact=True)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    training_script()

if __name__ == "__main__":
    main()