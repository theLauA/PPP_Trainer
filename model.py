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
X[np.isinf(X)] = 0
print(X.shape,y.shape,S.shape)
print(np.unique(S))

avgs = []
for forehand in [1,3,4,5]:
    for backhand in [101,102,103,104,105]:
        for smash in [201,202,203,204]:
            #print(forehand,backhand,smash)
            S_mask = np.logical_or(np.logical_or((S==forehand),(S==backhand)),(S==smash))

            X_train, X_test, y_train, y_test = X[~S_mask], X[S_mask],y[~S_mask], y[S_mask]

            #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



            #random-forest
            clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
            clf = clf.fit(X_train, y_train)
            #y_pred=clf.predict(X_test)

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

            y_pred = clf.predict_proba(X_test)
            y_pred_mask = np.max(y_pred,axis=1) < 0.4
            y_pred = np.argmax(y_pred,axis=1)
            y_pred[y_pred_mask] = 3
            #print(np.sum(y_pred_mask))
            #target_names = ['class 0', 'class 1', 'class 2','class 3']
            #print(classification_report(y_test, y_pred, target_names=target_names))
            avgs.append(accuracy_score(y_test, y_pred))
            #print(accuracy_score(y_test, y_pred))
avgs = np.array(avgs)
print(np.mean(avgs))