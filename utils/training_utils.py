from sklearn.base import BaseEstimator

from sklearn.preprocessing import label_binarize

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from typing import Literal, Callable

SEED = 314

def get_rf(seed=SEED):
    return RandomForestClassifier(max_features=4, random_state=seed)

def get_svc(seed=SEED):
    return SVC(probability=True, random_state=seed, decision_function_shape="ovr")

def get_nb(seed=SEED):
    return GaussianNB()

MODELS : dict[str, Callable[[], BaseEstimator]] = {
    "RandomForest": get_rf,
    "SVC": get_svc,
    "NaiveBayes": get_nb
}

def train_model(model_name : Literal["RandomForest", "SVC", "NaiveBayes"], params : dict, X, y, seed=SEED) -> BaseEstimator:
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in MODELS.")
    model = MODELS[model_name](seed=seed)
    model.set_params(**params)
    model.fit(X, y)
    return model

def test_model(model : BaseEstimator, X, y):
    y_pred = model.predict(X)
    accuracy = balanced_accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, zero_division=0)

    y_pred_proba = model.predict_proba(X)
    auc_roc = roc_auc_score(y, y_pred_proba, multi_class="ovr")

    y_bin = label_binarize(y, classes=list(range(30)))

    # Compute ROC curve for each class
    fpr = dict()
    tpr = dict()
    for i in range(30):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_pred_proba.ravel())
    
    return (accuracy, cm, report, auc_roc, fpr, tpr)

def cross_validate(model_name : Literal["RandomForest", "SVC", "NaiveBayes"], X, y, cv_folds=5):
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in MODELS.")
    model = MODELS[model_name]()
    scores = cross_val_score(model, X, y, n_jobs=-1, cv=cv_folds)
    return scores.mean(), scores.std()

def grid_search(model_name : Literal["RandomForest", "SVC", "NaiveBayes"], params : dict, X, y, cv_folds=5, seed=SEED):
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found in MODELS.")
    model = MODELS[model_name](seed=seed)
    grid_search = GridSearchCV(model, params, n_jobs=-1, cv=cv_folds, scoring="roc_auc_ovr")
    grid_search.fit(X, y)
    return (grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_) 