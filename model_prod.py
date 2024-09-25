# Importing necessary libraries
import random
from joblib import dump
import numpy as np
from sklearn.ensemble import IsolationForest

def model():
    # Setting up a random number generator with a seed for reproducibility
    rng = np.random.RandomState(100)

    # Generate random training data
    X = 0.3 * rng.randn(500, 1)
    X_train = np.r_[X + 2]
    X_train = np.round(X_train, 3)

    # Train the Isolation Forest model
    clf = IsolationForest(n_estimators=50, max_samples=500, random_state=rng, contamination=0.01)
    clf.fit(X_train)

    # Save the trained model to a file
    dump(clf, './isolation_forest.joblib')

if __name__ == "__main__":
    model()
