#
#   Code for the "Introduction to Machine Learning and Evolutonary Robotics" project.
#
from utils.data_utils import load_data, split_data
from utils.plot_utils import plot_confusion_matrix
from utils.training_utils import grid_search, cross_validate, train_model, test_model

import numpy as np
import random

RANDOM_FOREST_PARAMS = {
    'n_estimators': [10, 20, 30, 40, 50],
    'max_depth': [1, 3, 5, 10, 20]
}

SVC_PARAMS = {
    'C': [0.1, 0.5, 1.0, 10.0, 100.0, 1000.0],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
    'degree': [2, 3, 4, 5],
    'gamma': ['scale', 'auto']
}

CV_FOLDS = 5
TEST_RATIO = 0.2
SEED = 314

nb_params = {}

def training_script():
    X, y = load_data("dataset/leaf.csv")
    x_train, x_test, y_train, y_test = split_data(X, y, test_size=TEST_RATIO, random_state=SEED)
    print("Training Random Forest")
    _, best_rf_params, _ = grid_search("RandomForest", RANDOM_FOREST_PARAMS, x_train, y_train, cv_folds=CV_FOLDS)
    print(f"Best Random Forest Params: {best_rf_params}")
    random_forest_model = train_model("RandomForest", best_rf_params, x_train, y_train)
    rf_accuracy, rf_cm, rf_report, rf_auc = test_model(random_forest_model, x_test, y_test)
    print(f"Random Forest Accuracy: {rf_accuracy}")
    print(f"Random Forest Classification Report:\n {rf_report}")
    print(f"Random Forest AUC: {rf_auc}")
    plot_confusion_matrix(rf_cm, n_classes=30, plot_name="Random Forest")

    print("Training SVC")
    _, best_svc_params, _ = grid_search("SVC", SVC_PARAMS, X, y, cv_folds=CV_FOLDS)
    print(f"Best SVC Params: {best_svc_params}")
    svc_model = train_model("SVC", best_svc_params, x_train, y_train)
    svc_accuracy, svc_cm, svc_report, svc_auc = test_model(svc_model, x_test, y_test)
    print(f"SVC Accuracy: {svc_accuracy}")
    print(f"SVC Classification Report:\n {svc_report}")
    print(f"SVC AUC: {svc_auc}")
    plot_confusion_matrix(svc_cm, n_classes=30, plot_name="SVC")

    print("Training Naive Bayes")
    best_nb = cross_validate("NaiveBayes", X, y, cv_folds=CV_FOLDS)
    print(f"Average Naive Bayes Accuracy: {best_nb[0]}")
    nb_model = train_model("NaiveBayes", nb_params, x_train, y_train)
    nb_accuracy, nb_cm, nb_report, nb_auc = test_model(nb_model, x_test, y_test)
    print(f"Naive Bayes Accuracy: {nb_accuracy}")
    print(f"Naive Bayes Classification Report:\n {nb_report}")
    print(f"Naive Bayes AUC: {nb_auc}")
    plot_confusion_matrix(nb_cm, n_classes=30, plot_name="Naive Bayes")



def main():
    random.seed(SEED)
    np.random.seed(SEED)
    training_script()

if __name__ == "__main__":
    main()