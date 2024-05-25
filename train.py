import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from models.random_forest import RandomForest

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
    return X, y

def train_model():
    X, y = load_data('dataset/leaf.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    model = RandomForest(n_estimators=100, max_depth=None)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    cm = confusion_matrix(y_test, y_pred)
    print(f'Confusion Matrix:\n{cm}')
    model.save_model('models/random_forest.pkl')

    # Plotting the confusion matrix.
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.show()