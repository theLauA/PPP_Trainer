import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from prepare_data import prepare_data_4_classes,prepare_test_data
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(seed=42)

#prepare_data_4_classes()
prepare_test_data("./data/", "TestVideo", "right")

data = genfromtxt('features_4.csv', delimiter=',')

X_train, y_train, = data[:, :-2], data[:, -2]
X_train[np.isnan(X_train)] = 0
X_train[np.isinf(X_train)] = 0

data = genfromtxt('test_features_4.csv', delimiter=',')
X_test, y_test, = data[:, :-2], data[:, -2]
X_test[np.isnan(X_test)] = 0
X_test[np.isinf(X_test)] = 0

avgs_precision = []
avgs_recall = []
avgs_f1 = []


#random-forest
clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
clf = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

target_names = ['class 0', 'class 1', 'class 2','class 3']
print(classification_report(y_test, y_pred, labels=[0,1,2,3], target_names=target_names))
