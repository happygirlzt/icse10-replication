"""
Author: happygirlzt
Date: 16 Feb 2021
"""

# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# libSVM

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

X = ..
y = ..

clf.fit(X, y)

test_X
print(clf.predict(test_X))
