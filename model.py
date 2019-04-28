import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from prepare_data import prepare_data
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

np.random.seed(seed=42)

#prepare_data("./data/")
#data = genfromtxt('features.csv', delimiter=',')
data = genfromtxt('features_subject.csv', delimiter=',')
X, y, S = data[:, :-2], data[:, -2], data[:,-1]
X[np.isnan(X)] = 0
X[np.where(X==float("Inf"))] = 0
print(X.shape,y.shape,S.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



#random-forest
clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
clf = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

"""
#decision-tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

#logistic regression
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
clf=clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

#SVC
clf= SVC(gamma='scale')
clf=clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
"""


target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_test, y_pred, target_names=target_names))
print accuracy_score(y_test, y_pred)