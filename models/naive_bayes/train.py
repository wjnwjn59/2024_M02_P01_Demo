import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from .load_dataset import load_df, split_dataset
from .preprocess import preprocess_text, run_data_preprocess
from config.config import DatasetConfig


class SpamTextClfModel:
    def __init__(self, dictionary, le):
        self.model = GaussianNB()
        self.dictionary = dictionary
        self.le = le

    def evaluate(self, X_val, y_val):
        y_val_pred = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        return val_accuracy
    
    def train(self, X_train, y_train):
        print('Start training...')
        self.model.fit(X_train, y_train)
        print('Training completed!')

    def create_features(self, tokens):
        features = np.zeros(len(self.dictionary))

        for token in tokens:
            if token in self.dictionary:
                features[self.dictionary.index(token)] += 1

        features = np.array(features).reshape(1, -1)

        return features
    
    def predict(self, text):
        tokens = preprocess_text(text)
        features = self.create_features(tokens)
        prediction = self.model.predict(features)
        prediction_cls = self.le.inverse_transform(prediction)[0]

        return prediction_cls

def main():
    df = load_df(DatasetConfig.DATASET_PATH)
    X, y, dictionary, le = run_data_preprocess(df)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)
    model = SpamTextClfModel(dictionary=dictionary,
                             le=le)
    model.train(X_train=X_train,
                y_train=y_train)
    
    return model

model = main()