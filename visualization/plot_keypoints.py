import json
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import copy


def centering(body_points_arrays):
    nose_x = body_points_arrays[0]
    nose_y = body_points_arrays[1]
    for i in range(0, 25):
        body_points_arrays[i * 3] -= nose_x
        body_points_arrays[i * 3 + 1] -= nose_y
    return body_points_arrays


def get_body_point(body_points_arrays, num):
    return body_points_arrays[num * 3], body_points_arrays[num * 3 + 1]

def write_body_point(body_points_arrays, num, point_x, point_y):
    body_points_arrays[int(num) * 3] = point_x
    body_points_arrays[int(num) * 3 + 1] = point_y
    return body_points_arrays


def normalize(body_points_arrays):
    temp_array = copy.deepcopy(body_points_arrays)
    ideal_neck_length = 50.0
    nose_x, nose_y = get_body_point(body_points_arrays, 0)
    neck_x, neck_y = get_body_point(body_points_arrays, 1)
    print "nose: ", nose_x, " - ", nose_y
    print "neck: ", neck_x, " - ", neck_y
    neck_length = distance(nose_x, nose_y, neck_x, neck_y)
    ratio_to_divide = neck_length / ideal_neck_length

    for i in range(0, 25):
        point_x, point_y = get_body_point(body_points_arrays, i)
        angle_with_neck = angle(nose_x, nose_y, point_x, point_y)
        distance_with_neck = distance(nose_x, nose_y, point_x, point_y)
        x = nose_x + distance_with_neck/ratio_to_divide * math.cos(angle_with_neck)
        y = nose_y + distance_with_neck/ratio_to_divide * math.sin(angle_with_neck)
        temp_array = write_body_point(temp_array, i, x, y)
    return temp_array


def angle(center_x, center_y, touch_x, touch_y):
    delta_x = touch_x - center_x
    delta_y = touch_y - center_y
    theta_radians = math.atan2(delta_y, delta_x)
    return theta_radians


def distance(center_x, center_y, touch_x, touch_y):
    return ((center_x - touch_x) ** 2 + (center_y - touch_y) ** 2) ** 0.5


def normalize_coor(points):
    maxs = np.max(points, axis=0, keepdims=True)

    mins = np.min(points, axis=0, keepdims=True)
    return (points - mins) / (maxs - mins)


filepath = './data'
files = [os.path.join(filepath, file) for file in os.listdir(filepath)]

keypoints_pairs = [[0, 15], [0, 16], [15, 17], [16, 18], [0, 1],
                   [1, 2], [1, 5], [1, 8],
                   [2, 3], [3, 4],
                   [5, 6], [6, 7],
                   [8, 9], [8, 12],
                   [9, 10], [10, 11], [11, 22], [11, 24], [22, 23],
                   [12, 13], [13, 14], [14, 19], [14, 21], [19, 20]]

# keypoints_pairs = [[1,2],[1,0],[1,8],[1,5]]
keypoints_pairs = np.array(keypoints_pairs)

keypoints = [846.823, 167.054, 0.565449, 842.908, 249.32, 0.68593, 778.341, 255.217, 0.619857, 746.902, 343.37,
             0.669414, 758.76, 445.262, 0.67277, 903.709, 243.444, 0.654601, 946.674, 310.052, 0.674389, 923.264,
             337.539, 0.493888, 854.658, 445.235, 0.532447, 811.592, 447.171, 0.467015, 811.575, 580.397, 0.113247, 0,
             0, 0, 897.762, 443.28, 0.479745, 915.381, 572.557, 0.120189, 0, 0, 0, 835.029, 155.261, 0.585497, 856.605,
             153.296, 0.630708, 815.488, 165.033, 0.54106, 870.439, 165.049, 0.364095, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0]
keypoints = [778.25,196.388,0.947591,821.327,247.333,0.782251,805.769,255.219,0.672226,791.977,321.849,0.821075,725.404,304.099,0.734612,835.035,241.516,0.781059,819.351,284.564,0.749941,725.401,298.311,0.741568,872.279,435.449,0.665793,842.916,437.393,0.622784,827.245,535.352,0.132462,0,0,0,901.685,435.442,0.604647,899.762,549.071,0.122394,0,0,0,776.244,184.634,0.885253,792.091,184.629,0.891449,0,0,0,827.247,194.501,0.908301,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

keypoints = [315.465,108.118,0.874212,314.505,125.714,0.841814,292.945,129.592,0.883271,274.404,152.143,0.868033,289.079,160.971,0.774414,334.07,121.762,0.908957,339.917,145.292,0.749509,338.971,165.832,0.750317,317.442,173.626,0.683417,303.759,174.628,0.631987,308.618,198.078,0.0716132,0,0,0,331.149,170.696,0.640357,344.832,190.276,0.213628,0,0,0,309.566,105.159,0.975255,317.467,105.124,0.950778,300.835,105.192,0.97748,324.27,104.19,0.833174,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


keypoints = centering(keypoints)
keypoints = normalize(keypoints)

count = 0
keypoints_ls = []
for idx in range(0, len(keypoints), 3):
    # print(count,keypoints[idx],keypoints[idx+1],keypoints[idx+2])
    keypoints_ls.append([-keypoints[idx], -keypoints[idx + 1], keypoints[idx + 2]])

keypoints_ls = np.array(keypoints_ls)
keypoints_ls_normalize = keypoints_ls
# keypoints_ls_normalize = normalize_coor(keypoints_ls[:,:2])
plt.figure(figsize=(3, 8))
for pair in keypoints_pairs:
    k, l = pair
    if (keypoints_ls[k, 2] > 0 and keypoints_ls[l, 2] > 0):
        print(k, l, keypoints_ls_normalize[k, :2], keypoints_ls_normalize[l, :2])
        plt.plot([keypoints_ls_normalize[k, 0], keypoints_ls_normalize[l, 0]],
                 [keypoints_ls_normalize[k, 1], keypoints_ls_normalize[l, 1]])

plt.show()
