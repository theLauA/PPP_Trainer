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
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.feature_selection import chi2,f_classif,mutual_info_classif
import pickle
from prepare_data import read_all_frames, feature_extraction_wrapper

from imblearn.over_sampling import SMOTE

np.random.seed(seed=42)
######################################
# Test
######################################

#prepare_data_4_classes()
print("Start Testing")
#prepare_test_data("./data/", "TestVideo", "right")

data = genfromtxt('features_4.csv', delimiter=',')
#data = genfromtxt('features_subject.csv', delimiter=',')

X_train, y_train, = data[:, :-2], data[:, -2]
#X_train, y_train = X_train[y_train!=3], y_train[y_train!=3]
X_train[np.isnan(X_train)] = 0
X_train[np.isinf(X_train)] = 0

#X_train, y_train = X_train[y_train!=3], y_train[y_train!=3]
#SMOTE
#N = len(y_train[y_train==3]) * 1
#sm = SMOTE(random_state=13,ratio={0:N,1:N,2:N,3:N})
#X_train, y_train = sm.fit_sample(X_train, y_train)

print(y_train[y_train==0].shape,y_train[y_train==1].shape,y_train[y_train==2].shape,y_train[y_train==3].shape)
print(X_train.shape)
data = genfromtxt('test_features_4.csv', delimiter=',')
X_test, y_test, = data[:, :-1], data[:, -1]
X_test[np.isnan(X_test)] = 0
X_test[np.isinf(X_test)] = 0

avgs_precision = []
avgs_recall = []
avgs_f1 = []



#random-forest
# Train with All Training Data
clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)
clf = clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)

# Produce Label
y_pred_scores = clf.predict_proba(X_test)
y_pred = np.ones(X_test.shape[0])
y_pred *= 3
for i in range(1,len(y_pred)):
    if np.max(y_pred_scores[i]) > 0.4:
        y_pred[i] = np.argmax(y_pred_scores[i])
    elif y_pred[i] != y_pred[i-1]:
        y_pred[i] = 3

# Evaluate
target_names = ['class 0', 'class 1', 'class 2','class 3']
print(classification_report(y_test, y_pred, labels=[0,1,2,3], target_names=target_names))
print(confusion_matrix(y_test,y_pred))


# Scoring PipeLine
curr_start = 0
curr_class = 3
started = False
clf_0 = pickle.load(open('scorer_class_0','rb'))
clf_1 = pickle.load(open('scorer_class_1','rb'))
clf_2 = pickle.load(open('scorer_class_2','rb'))

frames = read_all_frames("./data/", "TestVideo", "right")
print("Start Scoring")
for idx in range(1,len(y_pred)):
    pred = y_pred[idx]
    if pred == y_pred[idx-1]:
        if started == False:
            started = True
            curr_start = idx
            curr_class =pred
    elif started:
        if curr_start != idx - 1 and idx - curr_start <= 5 and np.any(y_test[curr_start:idx]!=3):
            x = feature_extraction_wrapper(frames[(curr_start)*5:(idx-1)*5])
            x[np.isnan(x)] = 0
            x[np.isinf(x)] = 0
            x = x[np.newaxis,:]
            if curr_class == 0:
                print((curr_start)*5,(idx-1)*5,y_pred[curr_start:idx],y_test[curr_start:idx],clf_0.predict(x))
            elif curr_class == 1:
                print((curr_start)*5,(idx-1)*5,y_pred[curr_start:idx],y_test[curr_start:idx],clf_1.predict(x))
            elif curr_class == 2:
                print((curr_start)*5,(idx-1)*5,y_pred[curr_start:idx],y_test[curr_start:idx],clf_2.predict(x))
        started = False

'''
for i in range(len(y_pred)):
    if y_pred[i] != curr_class:
        if started:
            if y_test[i-1] != 3:
                print(curr_start,i-1,y_pred[curr_start:i],y_test[curr_start:i])
            started = False
        else:
            curr_start = i
            curr_class = y_pred[i]
            started = True
'''