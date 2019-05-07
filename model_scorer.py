import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from prepare_data import prepare_data_4_classes,prepare_data_4_classes_raw,prepare_data_scorer
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.feature_selection import chi2,f_classif,mutual_info_classif
import pickle
data = genfromtxt('features_4_score.csv', delimiter=',')
X, y, S = data[:, :-2], data[:, -2], data[:,-1]
X[np.isnan(X)] = 0
X[np.isinf(X)] = 0
print(X.shape,y.shape,S.shape)
print(np.unique(S))
print(y[(y==0)].shape,y[(y==1)].shape,y[(y==2)].shape,y[(y==3)].shape)

for i in [0,1,2]:
    
    y_preds = []
    y_tests = []

    X_r = X.copy()
    y_r = y.copy()

    y_r[y==i] = 100
    y_r[y!=i] = 0 
    
    for forehand in [1,3,4,5]:
        for backhand in [101,102,103,104,105]:
            for smash in [201,202,203,204]:


                #print(forehand,backhand,smash)
                S_mask = np.logical_or(np.logical_or((S==forehand),(S==backhand)),(S==smash))

                X_train, X_test, y_train, y_test = X_r[~S_mask], X_r[S_mask],y_r[~S_mask], y_r[S_mask]
                clf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)
                clf = clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)
                
                y_preds = np.append(y_preds,y_pred)
                y_tests = np.append(y_tests,y_test)
                #print(y_pred,y_test)
    
    #Save Scorer
    #clf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=0)
    #clf = clf.fit(X_r,y_r)
    #pickle.dump(clf,open("scorer_class_"+str(i),"wb"))

    print(i,  np.mean(np.sqrt((y_preds-y_tests)**2) ) )