import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

data = genfromtxt('../features.csv', delimiter=',')

for i in range(data.shape[1]):
    n_feature = 12
    keypoint = i//n_feature
    features_number = i%n_feature
    #Get Specific Features
    #if features_number == 8:
    if True:
        features_1 = data[data[:,-1] == 0,i]
        features_2 = data[data[:,-1] == 1,i]
        #features_3 = data[data[:,-1] == 2,i]

        plt.hist(features_1,bins=50,fc=(0,0,1,0.5))
        plt.hist(features_2,bins=50,fc=(1,0,0,0.5))
        #plt.hist(features_3,bins=50,fc=(0,1,0,0.5))
        plt.title("Feature #{} of Keypoint #{}".format(features_number,keypoint))
        plt.show()