from Dtree import DecisionTree
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


np.random.seed(45)
X, y = make_circles(n_samples=600, factor=0.1, noise=0.35, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = DecisionTree(max_depth=100)
clf.fit(X, y)

y_pred1 = clf.predict(X_test)

print("accuracy score of Bagging--->",accuracy_score(y_test,y_pred1))

