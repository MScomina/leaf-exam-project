import pickle
from sklearn.ensemble import RandomForestClassifier

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save_model(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, file_path):
        with open(file_path, 'rb') as file:
            self.model = pickle.load(file)