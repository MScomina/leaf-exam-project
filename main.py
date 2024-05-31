#
#   Code for the "Introduction to Machine Learning and Evolutonary Robotics" project.
#
from utils.data_utils import load_data, split_data
from utils.plot_utils import plot_confusion_matrix
from utils.training_utils import grid_search, cross_validate, train_model, test_model

import numpy as np
import random

RANDOM_FOREST_PARAMS = {
    'n_estimators': [10, 25, 50, 100, 200, 500, 1000],
    'max_depth': [1, 3, 5, 10, 25, 50]
}

SVC_PARAMS = {
    'C': [0.1, 0.5, 1.0, 10.0, 100.0, 1000.0],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

NB_PARAMS = {}

CV_FOLDS = 5
TEST_RATIO = 0.2
SEED = 314

def training_script(verbose=False):

    X, y = load_data("dataset/leaf.csv")
    x_train, x_test, y_train, y_test = split_data(X, y, test_size=TEST_RATIO, random_state=SEED)
    
    for model_name, model_params in zip(["RandomForest", "SVC", "NaiveBayes"], [RANDOM_FOREST_PARAMS, SVC_PARAMS, NB_PARAMS]):
        
        print(f"Training {model_name}")

        if model_name != "NaiveBayes":
            _, best_params, _ = grid_search(model_name, model_params, x_train, y_train, cv_folds=CV_FOLDS)
            print(f"Best {model_name} Params: {best_params}")
        else:
            best_params = {}
            best_nb = cross_validate(model_name, x_train, y_train, cv_folds=CV_FOLDS)
            print(f"Average {model_name} Accuracy: {best_nb[0]}")
        
        model = train_model(model_name, best_params, x_train, y_train)
        accuracy, cm, report, auc = test_model(model, x_test, y_test)
        print(f"{model_name} weighted Accuracy: {accuracy}")
        if verbose:
            print(f"{model_name} Classification Report:\n {report}")
        print(f"{model_name} AUC: {auc}")
        plot_confusion_matrix(cm, n_classes=30, plot_name=model_name)

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    training_script()

if __name__ == "__main__":
    main()