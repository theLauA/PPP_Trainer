import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from prepare_data import prepare_data
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.feature_selection import chi2,f_classif,mutual_info_classif
'''
#####################################################################
Dont Train with Class 3 (Not an action)
Test with Class 3 labels
#####################################################################
'''


np.random.seed(seed=42)

#prepare_data("./data/")
#data = genfromtxt('features.csv', delimiter=',')
data = genfromtxt('features_subject.csv', delimiter=',')
X, y, S = data[:, :-2], data[:, -2], data[:,-1]


data = genfromtxt('features_4.csv', delimiter=',')
X_4, y_4, S_4 = data[:, :-2], data[:, -2], data[:,-1]


X[np.isnan(X)] = 0
X[np.isinf(X)] = 0

X_4[np.isnan(X_4)] = 0
X_4[np.isinf(X_4)] = 0
print(X.shape,y.shape,S.shape)
print(np.unique(S))

avgs_precision = []
avgs_recall = []
avgs_f1 = []
print(y[(y==0)].shape,y[(y==1)].shape,y[(y==2)].shape)
cm = np.zeros((4,4))

sf = SelectPercentile(f_classif, percentile=10).fit(X, y)
X = sf.transform(X)
X_4 = sf.transform(X_4)

for forehand in [1,3,4,5]:
    for backhand in [101,102,103,104,105]:
        for smash in [201,202,203,204]:
            #print(forehand,backhand,smash)
            S_mask = np.logical_or(np.logical_or((S==forehand),(S==backhand)),(S==smash))
            S_mask_4 = np.logical_or(np.logical_or((S_4==forehand),(S_4==backhand)),(S_4==smash))

            X_train, X_test, y_train, y_test = X[~S_mask], X_4[S_mask_4],y[~S_mask], y_4[S_mask_4]

            #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
            #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



            #random-forest
            clf = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=0)
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

            y_pred_scores = clf.predict_proba(X_test)
            #y_pred_mask = np.max(y_pred,axis=1) < 0.5
            #y_pred = np.argmax(y_pred,axis=1)
            #y_pred[y_pred_mask] = 3
            #print(np.sum(y_pred_mask))
            y_pred = np.ones(X_test.shape[0])
            y_pred *= 3

            for i in range(1,len(y_pred)):
                if np.max(y_pred[i]) > 0.4:
                    y_pred[i] = np.argmax(y_pred_scores[i])
                if y_pred[i] != y_pred[i-1] and y_pred[i-1]!=3:
                    y_pred[i] = 3

            target_names = ['class 0', 'class 1', 'class 2','class 3']
            scores = classification_report(y_test, y_pred, labels=[0,1,2,3], target_names=target_names,output_dict=True)
            avgs_precision.append(scores["macro avg"]["precision"])
            avgs_recall.append(scores["macro avg"]["recall"])
            avgs_f1.append(scores["macro avg"]["f1-score"])
            cm += confusion_matrix(y_test,y_pred)
            #avgs.append(accuracy_score(y_test, y_pred))
            #print(accuracy_score(y_test, y_pred))
avgs_precision = np.array(avgs_precision)
avgs_recall = np.array(avgs_recall)
avgs_f1 = np.array(avgs_f1)

print(np.mean(avgs_precision),np.mean(avgs_recall),np.mean(avgs_f1))
print(cm)