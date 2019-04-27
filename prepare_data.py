import os
import json
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

def plot(keypoints):
    keypoints_pairs = [[0, 15], [0, 16], [15, 17], [16, 18], [0, 1],
                       [1, 2], [1, 5], [1, 8],
                       [2, 3], [3, 4],
                       [5, 6], [6, 7],
                       [8, 9], [8, 12],
                       [9, 10], [10, 11], [11, 22], [11, 24], [22, 23],
                       [12, 13], [13, 14], [14, 19], [14, 21], [19, 20]]
    keypoints_pairs = np.array(keypoints_pairs)
    count = 0
    keypoints_ls = []
    for idx in range(0, len(keypoints), 3):
        keypoints_ls.append([-keypoints[idx], -keypoints[idx + 1], keypoints[idx + 2]])

    keypoints_ls = np.array(keypoints_ls)
    keypoints_ls_normalize = keypoints_ls
    plt.figure(figsize=(3, 8))
    plt.yticks([-200, -180, -160, -140, -120, -100, -80, -60, -40, -20, 0, 20, 40])
    for pair in keypoints_pairs:
        k, l = pair
        if (keypoints_ls[k, 2] > 0 and keypoints_ls[l, 2] > 0):
            #print(k, l, keypoints_ls_normalize[k, :2], keypoints_ls_normalize[l, :2])
            plt.plot([keypoints_ls_normalize[k, 0], keypoints_ls_normalize[l, 0]],
                     [keypoints_ls_normalize[k, 1], keypoints_ls_normalize[l, 1]])

    plt.show()

def plots(keypoints_lists):
    keypoints_pairs = [[0, 15], [0, 16], [15, 17], [16, 18], [0, 1],
                       [1, 2], [1, 5], [1, 8],
                       [2, 3], [3, 4],
                       [5, 6], [6, 7],
                       [8, 9], [8, 12],
                       [9, 10], [10, 11], [11, 22], [11, 24], [22, 23],
                       [12, 13], [13, 14], [14, 19], [14, 21], [19, 20]]
    keypoints_pairs = np.array(keypoints_pairs)
    N = len(keypoints_lists) 
    for i in range(N):
        #ax = plt.subplot(1,N,i+1)
        #plt.setp(ax,yticks=range(-200,60,40))
        for pair in keypoints_pairs:
            k,l = pair
            if(keypoints_lists[i][k*3+2]>0 and keypoints_lists[i][l*3+2]>0):
                plt.plot([-keypoints_lists[i][k*3],-keypoints_lists[i][l*3]],
                        [-keypoints_lists[i][k*3+1],-keypoints_lists[i][l*3+1]])
    plt.show()
def get_body_point(body_points_arrays, num):
    return body_points_arrays[num * 3], body_points_arrays[num * 3 + 1]


def write_specific_point(body_points_arrays, num, point_x, point_y):
    body_points_arrays[int(num) * 3] = point_x
    body_points_arrays[int(num) * 3 + 1] = point_y
    return body_points_arrays


def angle(center_x, center_y, touch_x, touch_y):
    delta_x = touch_x - center_x
    delta_y = touch_y - center_y
    theta_radians = math.atan2(delta_y, delta_x)
    return theta_radians


def distance(center_x, center_y, touch_x, touch_y):
    return ((center_x - touch_x) ** 2 + (center_y - touch_y) ** 2) ** 0.5


def body_points_avg_x (body_points):
    count = 1
    total = 0
    for i in range(0, 25):
        if body_points[i * 3] !=0:
            total+=body_points[i * 3]
            count +=1
    return total/count


def get_body_points(filenames, focus_2ppl="none"):
    body_points_all_frame = []
    for filename in filenames:
        data = json.load(open(filename, "r"))  # read json files
        if focus_2ppl == "none":  # in case of 1ppl
            body_points_all_frame.append(data['people'][0]['pose_keypoints_2d'])
        else:  # in case of 2ppl
            if data['people'][1]["pose_keypoints_2d"] == [0]*75:
                body_points_all_frame.append(data['people'][0]['pose_keypoints_2d'])
                continue
            left_ppl, right_ppl = None, None
            if body_points_avg_x(data['people'][0]["pose_keypoints_2d"]) > body_points_avg_x(data['people'][1]["pose_keypoints_2d"]):  # compare nose's x values to see which is left
                left_ppl = data['people'][1]["pose_keypoints_2d"]
                right_ppl = data['people'][0]["pose_keypoints_2d"]
            else:
                left_ppl = data['people'][0]["pose_keypoints_2d"]
                right_ppl = data['people'][1]["pose_keypoints_2d"]
            body_points_all_frame.append(left_ppl) if focus_2ppl == "left" else body_points_all_frame.append(right_ppl)
    return body_points_all_frame


# centering with respect nose
def centering(body_points_arrays):
    for body_points_array in body_points_arrays:  # for each body points array
        nose_x = body_points_array[0]
        nose_y = body_points_array[1]
        for i in range(0, 25):
            body_points_array[i * 3] -= nose_x
            body_points_array[i * 3 + 1] -= nose_y
    return body_points_arrays


def normalize(body_points_arrays):
    ideal_neck_length = 50.0
    longest_neck = -1
    for body_points_array in body_points_arrays:  # find the longest neck among all frames
        nose_x, nose_y = get_specific_point(body_points_array, 0)
        neck_x, neck_y = get_specific_point(body_points_array, 1)
        neck_length = distance(nose_x,nose_y, neck_x, neck_y)
        if neck_length > longest_neck:
            longest_neck = neck_length
    ratio_to_divide = longest_neck / ideal_neck_length

    normalized = []
    for body_points_array in body_points_arrays: # normalize each body points array
        temp_array = copy.deepcopy(body_points_array)
        nose_x, nose_y = get_specific_point(body_points_array, 0)
        for i in range(0, 25):
            point_x, point_y = get_specific_point(body_points_array, i)
            angle_with_neck = angle(nose_x, nose_y, point_x, point_y)
            distance_with_neck = distance(nose_x, nose_y, point_x, point_y)
            x = nose_x + distance_with_neck / ratio_to_divide * math.cos(angle_with_neck)
            y = nose_y + distance_with_neck / ratio_to_divide * math.sin(angle_with_neck)
            temp_array = write_specific_point(temp_array, i, x, y)
        normalized.append(temp_array)
    return normalized


# def normalize_body_size(body_points_arrays, ideal_neck_size=20):


data_path = './data/'

lines_video_list = open(data_path + "video_list.csv", "r").readlines()
videos = [line_video_list.rstrip('\n').split(",")[0] for line_video_list in lines_video_list]

features = []
labels = []

for line_video_list in lines_video_list:  # for each video
    # basic info
    video_name = line_video_list.rstrip('\n').split(",")[0]
    ppl_focus = line_video_list.rstrip('\n').split(",")[1]

    # paths
    path_frame_range = data_path + "frame_range/" + video_name + ".csv"
    path_frame_jsons = data_path + "after_openpose/" + ("1ppl/" if ppl_focus == "none" else "2ppl/") + video_name

    # all frame jsons
    all_json_files = []
    for r, d, f in os.walk(path_frame_jsons):  # r=root, d=directories, f = files
        for file in f:
            if '.json' in file:
                all_json_files.append(os.path.join(r, file))
    print "******"
    print "Current Video: ", video_name
    lines_frame_range = open(path_frame_range, "r").readlines()

    for line_frame_range in lines_frame_range:  # for each action (frame range)

        frame_start = line_frame_range.rstrip('\n').split(",")[0]
        frame_end = line_frame_range.rstrip('\n').split(",")[1]
        print "frame start: ", frame_start, " - frame_end: ",frame_end
        action_frames = []
        for curr_json in all_json_files:
            if int(frame_start) <= int(curr_json.split("_")[-2]) <= int(frame_end):
                action_frames.append(curr_json)
        # FOR EACH FRAME, DO SOMETHING
        lol = get_body_points(action_frames, focus_2ppl=ppl_focus)
        print "lol length: ", len(lol)
        lol = centering(lol)
        old_lol = np.array(lol)
        lol = normalize(lol)
        for idx,lnl in enumerate(lol):
            #plot(lnl)
            plots( [lnl,old_lol[idx]] )
        # features.append("")
        labels.append(video_name[0])
