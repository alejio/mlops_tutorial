import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from joblib import dump

def train():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    X_train = train.drop(['subject', 'Activity'], axis=1)
    y_train = train.Activity
    X_test = test.drop(['subject', 'Activity'], axis=1)
    y_test = test.Activity

    parameters = {'max_depth':np.arange(2,10,2)}
    dt_classifier = DecisionTreeClassifier(max_depth=8)
    dt_classifier.fit(X_train, y_train)

    dump(dt_classifier, 'models/activity_classifier.joblib')

if __name__ == "__main__":
    train()