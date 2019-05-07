import numpy as np
from scipy.stats import linregress

#Expect Time Serie Points in shape [T,2]
def _extract_features_(points):

    result = np.zeros((1,0))
    result = np.append(result,_mean_(points))
    result = np.append(result,_mins_(points))
    result = np.append(result,_maxs_(points))
    result = np.append(result,_median_(points))
    result = np.append(result,_regress_(points))

    return result


def _mean_(points):
    return np.mean(points,axis=0)

def _mins_(points):
    return np.min(points,axis=0)

def _maxs_(points):
    return np.max(points,axis=0)

def _median_(points):
    return np.median(points,axis=0)

def _regress_(points):
    points -= points[0,:]
    slope_y, intercept_y, r_value_y, p_value_y, std_err_y = linregress(range(len(points)),points[:,1])
    slope_x, intercept_x, r_value_x, p_value_x, std_err_x = linregress(range(len(points)),points[:,0])
    slope, intercept, r_value, p_value, std_err = linregress(points[:,1],points[:,0])
    
    return np.array([slope_y, intercept_y, r_value_y, p_value_y, std_err_y,
                    slope_x, intercept_x, r_value_x, p_value_x, std_err_x,
                    slope, intercept, r_value, p_value, std_err])

def _extract_features_two_(points_1,points_2):

    dists = points_1 - points_2

    return _regress_(dists)