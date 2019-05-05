import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from prepare_data import prepare_data_4_classes,prepare_data_4_classes_raw
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.feature_selection import chi2,f_classif,mutual_info_classif

np.random.seed(seed=42)

#prepare_data_4_classes()
data = genfromtxt('features_4.csv', delimiter=',')
#prepare_data_4_classes_raw()
#data = genfromtxt('features_4_raw.csv',delimiter=',')
X, y, S = data[:, :-2], data[:, -2], data[:,-1]
X[np.isnan(X)] = 0
X[np.isinf(X)] = 0
print(X.shape,y.shape,S.shape)
print(np.unique(S))
print(y[(y==0)].shape,y[(y==1)].shape,y[(y==2)].shape,y[(y==3)].shape)


#X = SelectPercentile(f_classif, percentile=10).fit_transform(X, y)
#print(X.shape)

avgs_precision = []
avgs_recall = []
avgs_f1 = []

'''
# 7/3 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#random-forest
clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
clf = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

target_names = ['class 0', 'class 1', 'class 2','class 3']
print(classification_report(y_test, y_pred, labels=[0,1,2,3], target_names=target_names))
'''
cm = np.zeros((4,4))
counter = 0
features_imp = np.zeros(X.shape[1])
for forehand in [1,3,4,5]:
    for backhand in [101,102,103,104,105]:
        for smash in [201,202,203,204]:


            #print(forehand,backhand,smash)
            S_mask = np.logical_or(np.logical_or((S==forehand),(S==backhand)),(S==smash))

            X_train, X_test, y_train, y_test = X[~S_mask], X[S_mask],y[~S_mask], y[S_mask]

            #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



            #random-forest
            clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
            clf = clf.fit(X_train, y_train)
            y_pred=clf.predict(X_test)
            features_imp += clf.feature_importances_
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
            target_names = ['class 0', 'class 1', 'class 2','class 3']
            scores = classification_report(y_test, y_pred, labels=[0,1,2,3], target_names=target_names,output_dict=True)
            #print(classification_report(y_test, y_pred, labels=[0,1,2,3], target_names=target_names))
            avgs_precision.append(scores["macro avg"]["precision"])
            avgs_recall.append(scores["macro avg"]["recall"])
            avgs_f1.append(scores["macro avg"]["f1-score"])
            cm += confusion_matrix(y_test,y_pred)
            counter += 1
            #avgs.append(accuracy_score(y_test, y_pred))
            #print(accuracy_score(y_test, y_pred))
avgs_precision = np.array(avgs_precision)
avgs_recall = np.array(avgs_recall)
avgs_f1 = np.array(avgs_f1)

print(np.mean(avgs_precision),np.mean(avgs_recall),np.mean(avgs_f1))
print(cm/counter)
print(features_imp.argsort())
