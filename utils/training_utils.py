from sklearn.base import BaseEstimator

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from typing import Literal

MODELS : dict[str, BaseEstimator] = {
    "RandomForest": RandomForestClassifier,
    "SVC": SVC,
    "NaiveBayes": GaussianNB
}

def train_model(model_name : Literal["RandomForest", "SVC", "NaiveBayes"], params : dict, X, y):
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in MODELS.")
    if model_name == "SVC":
        model = MODELS[model_name](probability=True)
    else:
        model = MODELS[model_name]()
    model.set_params(**params)
    model.fit(X, y)
    return model

def test_model(model : BaseEstimator, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)

    y_pred_proba = model.predict_proba(X)
    auc_roc = roc_auc_score(y, y_pred_proba, multi_class="ovr")
    
    return (accuracy, cm, report, auc_roc)

def cross_validate(model_name : Literal["RandomForest", "SVC", "NaiveBayes"], X, y, cv_folds=5):
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in MODELS.")
    if model_name == "SVC":
        model = MODELS[model_name](probability=True)
    else:
        model = MODELS[model_name]()
    scores = cross_val_score(model, X, y, n_jobs=-1, cv=cv_folds)
    return scores.mean(), scores.std()

def grid_search(model_name : Literal["RandomForest", "SVC", "NaiveBayes"], params : dict, X, y, cv_folds=5):
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in MODELS.")
    if model_name == "SVC":
        model = MODELS[model_name](probability=True)
    else:
        model = MODELS[model_name]()
    grid_search = GridSearchCV(model, params, n_jobs=-1, cv=cv_folds)
    grid_search.fit(X, y)
    return (grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_) 