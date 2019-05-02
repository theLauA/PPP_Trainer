import os
import json
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import random
from features import _extract_features_

n_feature_keypoint = 23


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
            # print(k, l, keypoints_ls_normalize[k, :2], keypoints_ls_normalize[l, :2])
            plt.plot([keypoints_ls_normalize[k, 0], keypoints_ls_normalize[l, 0]],
                     [keypoints_ls_normalize[k, 1], keypoints_ls_normalize[l, 1]])

    plt.show()


def plots(keypoints_lists, filename=None):
    keypoints_pairs = [[0, 15], [0, 16], [15, 17], [16, 18], [0, 1],
                       [1, 2], [1, 5], [1, 8],
                       [2, 3], [3, 4],
                       [5, 6], [6, 7],
                       [8, 9], [8, 12],
                       [9, 10], [10, 11], [11, 22], [11, 24], [22, 23],
                       [12, 13], [13, 14], [14, 19], [14, 21], [19, 20]]
    keypoints_pairs = np.array(keypoints_pairs)
    N = len(keypoints_lists)

    plt.figure(figsize=(3, 8))
    for i in range(N):
        ax = plt.subplot(1, N, i + 1)

        for pair in keypoints_pairs:
            k, l = pair
            if (keypoints_lists[i][k * 3 + 2] > 0 and keypoints_lists[i][l * 3 + 2] > 0):
                ax.plot([-keypoints_lists[i][k * 3], -keypoints_lists[i][l * 3]],
                        [-keypoints_lists[i][k * 3 + 1], -keypoints_lists[i][l * 3 + 1]])
        ax.set_yticks(range(-200, 60, 40))
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.clf()


def plot_jittered(keypoints, probability):
    keypoints_pairs = [[0, 15], [0, 16], [15, 17], [16, 18], [0, 1],
                       [1, 2], [1, 5], [1, 8],
                       [2, 3], [3, 4],
                       [5, 6], [6, 7],
                       [8, 9], [8, 12],
                       [9, 10], [10, 11], [11, 22], [11, 24], [22, 23],
                       [12, 13], [13, 14], [14, 19], [14, 21], [19, 20]]
    keypoints_pairs = np.array(keypoints_pairs)

    plt.figure(figsize=(3, 8))
    plt.yticks([-200, -180, -160, -140, -120, -100, -80, -60, -40, -20, 0, 20, 40])
    for pair in keypoints_pairs:
        k, l = pair
        # print(k,l,keypoints[k],keypoints[l])
        if (probability[k] > 0 and probability[l] > 0):
            # print(k, l, keypoints_ls_normalize[k, :2], keypoints_ls_normalize[l, :2])
            plt.plot([keypoints[k, 0], keypoints[l, 0]],
                     [keypoints[k, 1], keypoints[l, 1]])

    plt.show()


def get_body25_specific(body25, num):
    return body25[num * 3], body25[num * 3 + 1]


def write_body25_specific(body25, num, point_x, point_y):
    body25[int(num) * 3] = point_x
    body25[int(num) * 3 + 1] = point_y
    return body25

def get_body25_exclude_confidence(body25):
    body25_exclude_confidence = []
    for i in range(0,25):
        x, y = get_body25_specific(body25, i)
        body25_exclude_confidence.append(x)
        body25_exclude_confidence.append(y)
    return body25_exclude_confidence


def angle(center_x, center_y, point_x, point_y):
    delta_x = point_x - center_x
    delta_y = point_y - center_y
    theta_radians = math.atan2(delta_y, delta_x)
    return theta_radians


def distance(center_x, center_y, point_x, point_y):
    return ((center_x - point_x) ** 2 + (center_y - point_y) ** 2) ** 0.5


# average x values
def body25_avg_x(body25):
    count = 1
    total = 0
    for i in range(0, 25):
        if body25[i * 3] != 0:
            total += body25[i * 3]
            count += 1
    return total / count


# difference in person between frames
def diff_in_frame(body25_1, body25_2):
    total = 0.0
    total_part = 1.0
    for i in range(0, 25):
        point_1_x, point_2_x = body25_1[i * 3], body25_2[i * 3]
        point_1_y, point_2_y = body25_1[i * 3 + 1], body25_2[i * 3 + 1]
        if point_1_x == 0 or point_1_y == 0 or point_2_x == 0 or point_2_y == 0:
            continue
        total += distance(point_1_x, point_1_y, point_2_x, point_2_y)
        total_part += 1
    return total/ total_part


# get body25s of frame(s)
def get_body25s(filenames, focus_2ppl="none"):
    body_points_all_frame = []
    for frame_json in filenames:
        json_data = json.load(open(frame_json, "r"))  # read json files
        ppl = get_ppl(json_data)
        if len(ppl) == 0:
            pass
        elif len(ppl) == 1:  # in case of 1ppl
            body_points_all_frame.append(ppl[0])
        else:  # in case of 2ppl
            left_ppl, right_ppl = ppl[0], ppl[1]
            body_points_all_frame.append(left_ppl) if focus_2ppl == "left" else body_points_all_frame.append(
                right_ppl)
    return body_points_all_frame


# get body25s of frame(s)
def get_body25s(filenames, focus_2ppl="none"):
    body_points_all_frame = []
    for frame_json in filenames:
        json_data = json.load(open(frame_json, "r"))  # read json files
        ppl = get_ppl(json_data)
        if len(ppl) == 0:
            body_points_all_frame.append([0]*75)
        elif len(ppl) == 1:  # in case of 1ppl
            body_points_all_frame.append(ppl[0])
        else:  # in case of 2ppl
            left_ppl, right_ppl = ppl[0], ppl[1]
            body_points_all_frame.append(left_ppl) if focus_2ppl == "left" else body_points_all_frame.append(
                right_ppl)
    return body_points_all_frame

# spine tracking variation of get_body25s
def get_body25s_spine_track(all_jsons, focus_2ppl="none"):
    # traverse all frames in the video
    body_25_all_frame = []
    previous_chosen = [0] * 75
    count = 0
    for curr_json in all_jsons:
        #print ("curr_json: ", curr_json)
        #print ("count: ", count)
        frame_data = json.load(open(curr_json, "r"))  # read json files
        ppl = get_ppl(frame_data)
        # Remove none person object
        temp_ppl = copy.deepcopy(ppl)
        for person in ppl:
            if get_spine_length(person) < 10:
                temp_ppl.remove(person)
                break
        ppl = copy.deepcopy(temp_ppl)

        # Selecting people
        if len(ppl) == 0:
            body_25_all_frame.append(body_25_all_frame[-1])
        elif len(ppl) == 1:
            body_25_all_frame.append(ppl[0])
        else:
            left_ppl, right_ppl = ppl[0], ppl[1]
            # original: compare the spines of each ppl with the ideal one and choose the least diffs
            # now: compare spines and choose the longer one, then compare diff in frames for last check
            left_spine = get_spine_length(left_ppl)
            right_spine = get_spine_length(right_ppl)
            if sum(previous_chosen) < 10:
                left_diff = right_diff = 0
            else:
                left_diff = diff_in_frame(left_ppl, previous_chosen)
                right_diff = diff_in_frame(right_ppl, previous_chosen)
            if left_spine <= right_spine and right_diff <= left_diff: # right longer, and smaller diff
                body_25_all_frame.append(right_ppl)
            elif left_spine <= right_spine and right_diff >= left_diff:
                body_25_all_frame.append(left_ppl)
            elif left_spine >= right_spine and right_diff >= left_diff: # left longer, and smaller diff
                body_25_all_frame.append(left_ppl)
            elif left_spine >= right_spine and right_diff <= left_diff:
                body_25_all_frame.append(right_ppl)

        previous_chosen = copy.deepcopy(body_25_all_frame[-1])
        count += 1
        # if count == 377:
        #    plot(previous_chosen)

    return body_25_all_frame


# return all people in the frame in a specific order
def get_ppl(frame_json):
    # in case of 0ppl
    if len(frame_json['people']) == 0:
        return []
    # in case of 1ppl
    if len(frame_json['people']) == 1:
        return [frame_json['people'][0]['pose_keypoints_2d']]
    # in case of 2ppl
    first_ppl = frame_json['people'][0]["pose_keypoints_2d"]
    second_ppl = frame_json['people'][1]["pose_keypoints_2d"]
    left_ppl, right_ppl = [0] * 75, [0] * 75
    if sum(first_ppl) == 0 and sum(second_ppl) == 0:  # both empty
        return []
    elif sum(first_ppl) == 0:  # one's empty
        return [second_ppl]
    elif sum(second_ppl) == 0:  # vice versa
        return [first_ppl]
    else:  # both not empty
        if body25_avg_x(first_ppl) > body25_avg_x(second_ppl):
            left_ppl = second_ppl
            right_ppl = first_ppl
        else:
            left_ppl = first_ppl
            right_ppl = second_ppl
    return [left_ppl, right_ppl]


# length from neck to mid hip
def get_spine_length(body25):
    neck_x, neck_y = get_body25_specific(body25, 1)
    hip_x, hip_y = get_body25_specific(body25, 8)
    if neck_x == 0 and neck_y == 0:
        return 0
    if hip_x == 0 and hip_y == 0:
        return 0
    spine_length = distance(neck_x, neck_y, hip_x, hip_y)
    return spine_length

# no use for now
def get_ideal_spine_length(all_jsons, spine_frame_number, focus_2ppl):
    goal_spine_length = -1

    # traverse all json file til reach the frame_spine frame, and get spine length
    for curr_json in all_jsons:
        if int(curr_json.split("_")[-2]) == spine_frame_number:
            frame_spine = json.load(open(curr_json, "r"))
            ppl = get_ppl(frame_spine)
            if focus_2ppl == "left" or focus_2ppl == "none":
                ppl_to_track = ppl[0]
            else:
                ppl_to_track = ppl[1]
            goal_spine_length = get_spine_length(ppl_to_track)
            break
    return goal_spine_length

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
        nose_x, nose_y = get_body25_specific(body_points_array, 0)
        neck_x, neck_y = get_body25_specific(body_points_array, 1)
        neck_length = distance(nose_x, nose_y, neck_x, neck_y)
        if neck_length > longest_neck:
            longest_neck = neck_length
    ratio_to_divide = longest_neck / ideal_neck_length

    normalized = []
    for body_points_array in body_points_arrays:  # normalize each body points array
        temp_array = copy.deepcopy(body_points_array)
        nose_x, nose_y = get_body25_specific(body_points_array, 0)
        for i in range(0, 25):
            point_x, point_y = get_body25_specific(body_points_array, i)
            angle_with_neck = angle(nose_x, nose_y, point_x, point_y)
            distance_with_neck = distance(nose_x, nose_y, point_x, point_y)
            if ratio_to_divide * math.cos(angle_with_neck) != 0:
                x = nose_x + distance_with_neck / ratio_to_divide * math.cos(angle_with_neck)
            else:
                x = nose_x
            if ratio_to_divide * math.sin(angle_with_neck) != 0:
                y = nose_y + distance_with_neck / ratio_to_divide * math.sin(angle_with_neck)
            else:
                y = nose_y
            temp_array = write_body25_specific(temp_array, i, x, y)
        normalized.append(temp_array)
    return normalized


# Change flat keypoints shape
def normalize_range(keypoints):
    keypoints_ls = []
    for idx in range(0, len(keypoints), 3):
        keypoints_ls.append([-keypoints[idx], -keypoints[idx + 1], keypoints[idx + 2]])
    keypoints_ls = np.array(keypoints_ls)
    maxs = np.max(keypoints_ls, axis=0, keepdims=True)
    mins = np.min(keypoints_ls, axis=0, keepdims=True)
    return (keypoints_ls - mins) / (maxs - mins)


def prepare_data(data_path):
    # data_path = './data/'

    lines_video_list = open(data_path + "video_list.csv", "r").readlines()
    videos = [line_video_list.rstrip('\n').split(",")[0] for line_video_list in lines_video_list]

    features = []
    labels = []
    videos = []
    for line_video_list in lines_video_list:  # for each video
        # basic info
        video_name = line_video_list.rstrip('\n').split(",")[0]
        ppl_focus = line_video_list.rstrip('\n').split(",")[1]

        # paths
        path_frame_range = data_path + "frame_range/" + video_name + ".csv"
        path_frame_jsons = data_path + "body_25/" + ("1ppl/" if ppl_focus == "none" else "2ppl/") + video_name

        # all frame jsons
        all_json_files = []
        for r, d, f in os.walk(path_frame_jsons):  # r=root, d=directories, f = files
            for file in f:
                if '.json' in file:
                    all_json_files.append(os.path.join(r, file))
        print("******")
        print("Current Video: ", video_name)
        lines_frame_range = open(path_frame_range, "r").readlines()

        for line_frame_range in lines_frame_range:  # for each action (frame range)

            frame_start = line_frame_range.rstrip('\n').split(",")[0]
            frame_end = line_frame_range.rstrip('\n').split(",")[1]
            print("frame start: ", frame_start, " - frame_end: ", frame_end)
            action_frames = []
            for curr_json in all_json_files:
                if int(frame_start) <= int(curr_json.split("_")[-2]) <= int(frame_end):
                    action_frames.append(curr_json)
            # FOR EACH FRAME, DO SOMETHING
            lol = get_body25s(action_frames, focus_2ppl=ppl_focus)
            print("lol length: ", len(lol))
            lol = centering(lol)
            old_lol = np.array(lol)
            lol = normalize(lol)

            points = []
            for idx, lnl in enumerate(lol):
                # plots( [lnl,old_lol[idx]] )
                points.append(normalize_range(lnl))
            points = np.array(points)

            N, K, dim = points.shape
            # N = number of frames, K = num_body_points, dim = 3 points

            window = 20
            step = 5

            for w in range(0, N, step):
                current_window = points[w:w + window, :, :]  # [0:20,25,3]
                current_features = np.zeros(K * n_feature_keypoint)

                n, nx, ny = current_window.shape  # [0:20,25,3]
                if n < 10:  # less than 10 frames in the window
                    break

                for i in range(K):
                    # Get Specific points over the Window
                    point_time_series = current_window[:, i, :].reshape(n, ny)
                    # Remove Invalid points and Drop Probability
                    point_time_series = point_time_series[point_time_series[:, 2] > 0]
                    point_time_series = point_time_series[:, 0:2]

                    if point_time_series.shape[0] < 10:
                        # Just Leave features as Zeros for such keypoint
                        break

                    # print(point_time_series.shape)
                    current_features[
                    i * n_feature_keypoint:i * n_feature_keypoint + n_feature_keypoint] = _extract_features_(
                        point_time_series)

                features.append(current_features)
                labels.append(int(video_name[0]))
                videos.append(int(video_name[0:3]))
    features = np.array(features)
    labels = np.array(labels)
    videos = np.array(videos)
    print(features.shape, labels.shape, videos.shape)
    np.savetxt("features_subject.csv",
               np.append(np.append(features, labels[:, np.newaxis], axis=1), videos[:, np.newaxis], axis=1),
               delimiter=",")


def prepare_test_data(data_path, test_video_name, side , file_path="test_features_4.csv"):
    features = []
    labels = []

    # path to get frame in jsons
    path_frame_jsons = data_path + 'body_25/test/' + test_video_name + '/'
    # path to get frame range labels
    path_frame_range = data_path + 'frame_range/' + test_video_name + '.csv'
    # get all frames' body 25
    path_all_frame_jsons = []
    for r, d, f in os.walk(path_frame_jsons):  # r=root, d=directories, f = files
        for file in f:
            if '.json' in file:
                path_all_frame_jsons.append(os.path.join(r, file))
    all_frame_jsons = get_body25s_spine_track(path_all_frame_jsons,focus_2ppl=side)
    all_frame_jsons = centering(all_frame_jsons)
    all_frame_jsons = np.array(all_frame_jsons)
    all_frame_jsons = normalize(all_frame_jsons)

    points = []
    for idx, lnl in enumerate(all_frame_jsons):
        # plots( [lnl,all_frame_jsons[idx]] )
        points.append(normalize_range(lnl))
    points = np.array(points)

    videos = []
    # pre label
    lines_frame_range = open(path_frame_range, "r").readlines()
    ranges_to_label = np.ones(len(all_frame_jsons))
    ranges_to_label *= 3
    for line_frame_range in lines_frame_range:
        # Find label for Each Frame
        frame_start = int(line_frame_range.rstrip('\n').split(",")[0])
        frame_end = int(line_frame_range.rstrip('\n').split(",")[1])
        action_label = int(line_frame_range.rstrip('\n').split(",")[2])
        ranges_to_label[frame_start:frame_end + 1] = action_label
        print(frame_start, frame_end, action_label)
    # *********************************
    window_size = 20
    window_step = 5

    for window_start_idx in range(0, len(all_frame_jsons) - window_size, window_step):

        window_start_label = ranges_to_label[window_start_idx]
        window_end_label = ranges_to_label[window_start_idx + window_size - 1]

        # if all label in the window equal to window start
        if np.all(ranges_to_label[window_start_idx:window_start_idx + window_size] == window_start_label):
            # print(window_start_idx,ranges_to_label[window_start_idx:window_start_idx+window_size])
            labels.append(window_start_label)
        elif window_start_label == window_end_label:  # if window start label = window end label
            continue
        else:
            labels.append(3)

        curr_window_points = np.array(points[window_start_idx:window_start_idx + window_size, :, :])
        N, K, dim = curr_window_points.shape
        current_features = np.zeros(K * n_feature_keypoint)
        for i in range(K):
            # Get Specific points over the Window
            n, nx, ny = curr_window_points.shape
            point_time_series = curr_window_points[:, i, :].reshape(n, ny)
            # Remove Invalid points and Drop Probability
            point_time_series = point_time_series[point_time_series[:, 2] > 0]
            point_time_series = point_time_series[:, 0:2]
            if point_time_series.shape[0] < 10:
                break
            current_features[
            i * n_feature_keypoint:i * n_feature_keypoint + n_feature_keypoint] = _extract_features_(
                point_time_series)
        features.append(current_features)
    features = np.array(features)
    labels = np.array(labels)
    print(features.shape, labels.shape)
    np.savetxt(file_path, np.append(features, labels[:, np.newaxis], axis=1),delimiter=",")


def prepare_data_4_class_dnn(data_path='./data/', file_path="dnn_features_4.csv"):

    lines_video_list = open(data_path + "video_list.csv", "r").readlines()

    features = []
    labels = []
    count = 0

    for line_video_list in lines_video_list:  # for each video
        # basic info
        video_name = line_video_list.rstrip('\n').split(",")[0]
        ppl_focus = line_video_list.rstrip('\n').split(",")[1]

        # paths
        path_frame_range = data_path + "frame_range/" + video_name + ".csv"
        path_frame_jsons = data_path + "body_25/" + ("1ppl/" if ppl_focus == "none" else "2ppl/") + video_name

        # all frame jsons
        all_json_files = []
        for r, d, f in os.walk(path_frame_jsons):  # r=root, d=directories, f = files
            for file in f:
                if '.json' in file:
                    all_json_files.append(os.path.join(r, file))
        print("******")
        print("Current Video: ", video_name)
        lines_frame_range = open(path_frame_range, "r").readlines()

        ranges_to_label = np.ones(len(all_json_files))
        ranges_to_label *= 3

        for line_frame_range in lines_frame_range:
            # Find label for Each Frame
            frame_start = int(line_frame_range.rstrip('\n').split(",")[0])
            frame_end = int(line_frame_range.rstrip('\n').split(",")[1])
            ranges_to_label[frame_start:frame_end + 1] = int(video_name[0])
            print(frame_start, frame_end, int(video_name[0]))

        lol = get_body25s(all_json_files, focus_2ppl=ppl_focus)
        count += len(lol)
        lol = centering(lol)
        lol = normalize(lol)
        for i in range(len(ranges_to_label)):
            body_pt = get_body25_exclude_confidence(lol[i])
            if (sum(body_pt) != 0):
                features.append(body_pt)
                labels.append(ranges_to_label[i])
    features = np.array(features)
    labels = np.array(labels)
    print(count)
    print(features.shape, labels.shape)
    np.savetxt(file_path, np.append(features, labels[:, np.newaxis], axis=1),delimiter=",")


def prepare_test_data_dnn(data_path, test_video_name, side, file_path="dnn_test_features_4.csv"):
    features = []
    labels = []

    # path to get frame in jsons
    path_frame_jsons = data_path + 'body_25/test/' + test_video_name + '/'
    # path to get frame range labels
    path_frame_range = data_path + 'frame_range/' + test_video_name + '.csv'
    # get all frames' body 25
    path_all_frame_jsons = []
    for r, d, f in os.walk(path_frame_jsons):  # r=root, d=directories, f = files
        for file in f:
            if '.json' in file:
                path_all_frame_jsons.append(os.path.join(r, file))
    all_frame_jsons = get_body25s_spine_track(path_all_frame_jsons,focus_2ppl=side)
    all_frame_jsons = centering(all_frame_jsons)
    all_frame_jsons = np.array(all_frame_jsons)
    all_frame_jsons = normalize(all_frame_jsons)

    # pre label
    lines_frame_range = open(path_frame_range, "r").readlines()
    ranges_to_label = np.ones(len(all_frame_jsons))
    ranges_to_label *= 3
    for line_frame_range in lines_frame_range:
        # Find label for Each Frame
        frame_start = int(line_frame_range.rstrip('\n').split(",")[0])
        frame_end = int(line_frame_range.rstrip('\n').split(",")[1])
        action_label = int(line_frame_range.rstrip('\n').split(",")[2])
        ranges_to_label[frame_start:frame_end + 1] = action_label
        print(frame_start, frame_end, action_label)
    # *********************************
    for i in range(len(all_frame_jsons)):
        body_pt = get_body25_exclude_confidence(all_frame_jsons[i])
        features.append(body_pt)
        labels.append(ranges_to_label[i])
    features = np.array(features)
    labels = np.array(labels)
    print(features.shape, labels.shape)
    np.savetxt(file_path, np.append(features, labels[:, np.newaxis], axis=1),delimiter=",")

def prepare_data_4_classes(data_path="./data/", file_path="features_4.csv"):
    # data_path = './data/'

    lines_video_list = open(data_path + "video_list.csv", "r").readlines()
    videos = [line_video_list.rstrip('\n').split(",")[0] for line_video_list in lines_video_list]

    features = []
    labels = []
    videos = []
    for line_video_list in lines_video_list:  # for each video
        # basic info
        video_name = line_video_list.rstrip('\n').split(",")[0]
        ppl_focus = line_video_list.rstrip('\n').split(",")[1]

        # paths
        path_frame_range = data_path + "frame_range/" + video_name + ".csv"
        path_frame_jsons = data_path + "body_25/" + ("1ppl/" if ppl_focus == "none" else "2ppl/") + video_name

        # all frame jsons
        all_json_files = []
        for r, d, f in os.walk(path_frame_jsons):  # r=root, d=directories, f = files
            for file in f:
                if '.json' in file:
                    all_json_files.append(os.path.join(r, file))
        print("******")
        print("Current Video: ", video_name)
        lines_frame_range = open(path_frame_range, "r").readlines()

        ranges_to_label = np.ones(len(all_json_files))
        ranges_to_label *= 3
        for line_frame_range in lines_frame_range:
            # Find label for Each Frame
            frame_start = int(line_frame_range.rstrip('\n').split(",")[0])
            frame_end = int(line_frame_range.rstrip('\n').split(",")[1])
            ranges_to_label[frame_start:frame_end + 1] = int(video_name[0])
            print(frame_start, frame_end, int(video_name[0]))

        window_size = 20
        window_step = 5

        for window_start_idx in range(0, len(all_json_files) - window_size, window_step):

            window_start_label = ranges_to_label[window_start_idx]
            window_end_label = ranges_to_label[window_start_idx + window_size - 1]
            # if all label in the window equal to window start
            if np.all(ranges_to_label[window_start_idx:window_start_idx + window_size] == window_start_label):
                # print(window_start_idx,ranges_to_label[window_start_idx:window_start_idx+window_size])
                labels.append(window_start_label)
            elif window_start_label == window_end_label: # if window start label = window end label
                continue
            else:

                labels.append(3)

            action_frames = all_json_files[window_start_idx:window_start_idx + window_size]

            lol = get_body25s(action_frames, focus_2ppl=ppl_focus)
            lol = centering(lol)
            lol = normalize(lol)

            points = []
            for lnl in lol:
                points.append(normalize_range(lnl))
            curr_window_points = np.array(points)
            N, K, dim = curr_window_points.shape
            current_features = np.zeros(K * n_feature_keypoint)
            for i in range(K):
                # Get Specific points over the Window
                n, nx, ny = curr_window_points.shape
                point_time_series = curr_window_points[:, i, :].reshape(n, ny)
                # Remove Invalid points and Drop Probability
                point_time_series = point_time_series[point_time_series[:, 2] > 0]
                point_time_series = point_time_series[:, 0:2]
                if point_time_series.shape[0] < 10:
                    break
                current_features[
                i * n_feature_keypoint:i * n_feature_keypoint + n_feature_keypoint] = _extract_features_(
                    point_time_series)
            features.append(current_features)
            videos.append(int(video_name[0:3]))
    features = np.array(features)
    labels = np.array(labels)
    videos = np.array(videos)
    print(features.shape, labels.shape, videos.shape)
    np.savetxt(file_path, np.append(np.append(features, labels[:, np.newaxis], axis=1), videos[:, np.newaxis], axis=1),
               delimiter=",")


def prepare_data_as_figure(data_path="./data/"):
    # data_path = './data/'

    lines_video_list = open(data_path + "video_list.csv", "r").readlines()
    # videos = [line_video_list.rstrip('\n').split(",")[0] for line_video_list in lines_video_list]

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
        print("******")
        print("Current Video: ", video_name)
        lines_frame_range = open(path_frame_range, "r").readlines()

        ranges_to_label = np.ones(len(all_json_files))
        ranges_to_label *= 3
        for line_frame_range in lines_frame_range:
            # Find label for Each Frame
            frame_start = int(line_frame_range.rstrip('\n').split(",")[0])
            frame_end = int(line_frame_range.rstrip('\n').split(",")[1])
            ranges_to_label[frame_start:frame_end + 1] = int(video_name[0])
            print(frame_start, frame_end, int(video_name[0]))

            lol = get_body25s(all_json_files, focus_2ppl=ppl_focus)
            lol = centering(lol)
            lol = normalize(lol)

            points = []
            for idx, lnl in enumerate(lol):
                # pass
                plots([lnl], './figures/' + video_name + "_" + str(idx) + "_" + str(int(ranges_to_label[idx])))
                points.append(normalize_range(lnl))


def prepare_data_4_classes_raw(data_path="./data/", file_path="features_4_raw.csv"):
    # data_path = './data/'

    lines_video_list = open(data_path + "video_list.csv", "r").readlines()
    videos = [line_video_list.rstrip('\n').split(",")[0] for line_video_list in lines_video_list]

    features = []
    labels = []
    videos = []
    for line_video_list in lines_video_list:  # for each video
        # basic info
        video_name = line_video_list.rstrip('\n').split(",")[0]
        ppl_focus = line_video_list.rstrip('\n').split(",")[1]

        # paths
        path_frame_range = data_path + "frame_range/" + video_name + ".csv"
        path_frame_jsons = data_path + "body_25/" + ("1ppl/" if ppl_focus == "none" else "2ppl/") + video_name

        # all frame jsons
        all_json_files = []
        for r, d, f in os.walk(path_frame_jsons):  # r=root, d=directories, f = files
            for file in f:
                if '.json' in file:
                    all_json_files.append(os.path.join(r, file))
        print("******")
        print("Current Video: ", video_name)
        lines_frame_range = open(path_frame_range, "r").readlines()

        ranges_to_label = np.ones(len(all_json_files))
        ranges_to_label *= 3
        for line_frame_range in lines_frame_range:
            # Find label for Each Frame
            frame_start = int(line_frame_range.rstrip('\n').split(",")[0])
            frame_end = int(line_frame_range.rstrip('\n').split(",")[1])
            ranges_to_label[frame_start:frame_end + 1] = int(video_name[0])
            print(frame_start, frame_end, int(video_name[0]))

        window_size = 20
        window_step = 5

        for window_start_idx in range(0, len(all_json_files) - window_size, window_step):
            current_label = 3
            window_start_label = ranges_to_label[window_start_idx]
            window_end_label = ranges_to_label[window_start_idx + window_size - 1]
            if np.all(ranges_to_label[window_start_idx:window_start_idx + window_size] == window_start_label):
                # print(window_start_idx,ranges_to_label[window_start_idx:window_start_idx+window_size])
                current_label = window_start_label
            elif window_start_label == window_end_label:
                continue

            action_frames = all_json_files[window_start_idx:window_start_idx + window_size]

            lol = get_body25s(action_frames, focus_2ppl=ppl_focus)
            lol = centering(lol)
            lol = normalize(lol)

            points = []
            for lnl in lol:
                points.append(normalize_range(lnl))
            curr_window_points = np.array(points)
            if curr_window_points[:, :, 0:2].flatten()[np.newaxis, :].shape[1] != 1000:
                print(window_start_idx)
                continue
            features.append(curr_window_points[:, :, 0:2].flatten())
            videos.append(int(video_name[0:3]))
            labels.append(current_label)
    features = np.array(features)
    labels = np.array(labels)
    videos = np.array(videos)
    print(features.shape, labels.shape, videos.shape)
    np.savetxt(file_path, np.append(np.append(features, labels[:, np.newaxis], axis=1), videos[:, np.newaxis], axis=1),
               delimiter=",")


if __name__ == "__main__":
    # prepare_data("./data/")
    # prepare_test_data("./data/", "TestVideo", "right")
    prepare_data_4_class_dnn()
    prepare_test_data_dnn("./data/", "TestVideo", "right")
    # prepare_data_4_classes()
    # prepare_data_4_classes_raw()
    # prepare_data_as_figure("./data/")`