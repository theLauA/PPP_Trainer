import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from prepare_data import prepare_data, prepare_data_4_classes
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.feature_selection import chi2,f_classif,mutual_info_classif
from imblearn.over_sampling import SMOTE

'''
#####################################################################
Train with Full Video with SMOTE
Validation + Test
#####################################################################
'''


np.random.seed(seed=42)

#prepare_data("./data/")
#prepare_data_4_classes()
#data = genfromtxt('features.csv', delimiter=',')
#data = genfromtxt('features_subject.csv', delimiter=',')
#X, y, S = data[:, :-2], data[:, -2], data[:,-1]


data = genfromtxt('features_4.csv', delimiter=',')
X_4, y_4, S_4 = data[:, :-2], data[:, -2], data[:,-1]


X_4[np.isnan(X_4)] = 0
X_4[np.isinf(X_4)] = 0

S = S_4.copy()
X, y = X_4.copy(), y_4.copy()


print(X.shape,y.shape,S.shape)
print(np.unique(S))

avgs_precision = []
avgs_recall = []
avgs_f1 = []
print(y_4[(y_4==0)].shape,y_4[(y_4==1)].shape,y_4[(y_4==2)].shape,y_4[(y_4==3)].shape)
cm = np.zeros((4,4))


for forehand in [1,3,4,5]:
    for backhand in [101,102,103,104,105]:
        for smash in [201,202,203,204]:
            #print(forehand,backhand,smash)
            S_mask = np.logical_or(np.logical_or((S==forehand),(S==backhand)),(S==smash))
            
            X_train, X_test, y_train, y_test = X[~S_mask], X[S_mask],y[~S_mask], y[S_mask]

                        
            #SMOTE
            N = len(y_train[y_train==3]) * 1
            sm = SMOTE(random_state=13,ratio={0:N,1:N,2:N,3:N})
            X_train, y_train = sm.fit_sample(X_train, y_train)
            
            print(y_train[y_train==0].shape,y_train[y_train==1].shape,y_train[y_train==2].shape,y_train[y_train==3].shape)

            #random-forest
            clf = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=0)
            clf = clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)

            target_names = ['class 0', 'class 1', 'class 2','class 3']
            scores = classification_report(y_test, y_pred, labels=[0,1,2,3], target_names=target_names,output_dict=True)
            avgs_precision.append(scores["macro avg"]["precision"])
            avgs_recall.append(scores["macro avg"]["recall"])
            avgs_f1.append(scores["macro avg"]["f1-score"])
            cm += confusion_matrix(y_test,y_pred)
            
avgs_precision = np.array(avgs_precision)
avgs_recall = np.array(avgs_recall)
avgs_f1 = np.array(avgs_f1)

print(np.mean(avgs_precision),np.mean(avgs_recall),np.mean(avgs_f1))
print(cm)


# Test Time
X_train, y_train, = data[:, :-2], data[:, -2]
#X_train, y_train = X_train[y_train!=3], y_train[y_train!=3]
X_train[np.isnan(X_train)] = 0
X_train[np.isinf(X_train)] = 0

#X_train, y_train = X_train[y_train!=3], y_train[y_train!=3]
#SMOTE
N = len(y_train[y_train==3]) * 1
sm = SMOTE(random_state=13,ratio={0:N,1:N,2:N,3:N})
X_train, y_train = sm.fit_sample(X_train, y_train)

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