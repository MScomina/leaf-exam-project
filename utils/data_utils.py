import pandas as pd
import pickle

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

LABELS_METADATA = [
    'Class',
    'Specimen Number'
]

LABELS_SHAPE = [
    'Eccentricity',
    'Aspect Ratio',
    'Elongation',
    'Solidity',
    'Stochastic Convexity',
    'Isoperimetric Factor',
    'Maximal Indentation Depth'
]

LABELS_TEXTURE = [
    'Lobedness',
    'Average Intensity',
    'Average Contrast',
    'Smoothness',
    'Third moment',
    'Uniformity',
    'Entropy'
]

def load_data(file_path):
    # Loading the dataset with the labels from the PDF.
    data = pd.read_csv(file_path, names=LABELS_METADATA + LABELS_SHAPE + LABELS_TEXTURE)
    data.drop('Specimen Number', axis=1, inplace=True)  # Dropping the 'Specimen Number' column, as it is not useful for training and is, instead, just metadata.
    X = data.drop('Class', axis=1)
    y = data['Class']
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y

def save_model(model : BaseEstimator, file_path) -> None:
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(file_path) -> BaseEstimator:
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

def split_data(X, y, test_size=0.2, random_state=None):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)