import numpy as np
from scipy.stats import linregress

#Expect Time Serie Points in shape [T,2]
def _extract_features(points):

    return


def _mean_(points):
    return np.mean(points,axis=0)

def _mins_(points):
    return np.min(points,axis=0)

def _maxs_(points):
    return np.max(points,axis=0)

def _median_(points):
    return np.median(points,axis=0)

def _regress_(points):
    slope, intercept, r_value, p_value, std_err = linregress(points[:,0],points[:,1])
    return np.array([slope, intercept, r_value, p_value, std_err])

