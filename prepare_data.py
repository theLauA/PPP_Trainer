import os
import json
import math

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

def get_body_points(filenames, focus_2ppl="none"):
    body_points_all_frame = []
    for filename in filenames:
        data = json.load(open(filename,"r"))  # read json files
        if focus_2ppl == "none":  # in case of 1ppl
            body_points_all_frame.append(data['people'][0]['pose_keypoints_2d'])
        else:   # in case of 2ppl
            left_ppl, right_ppl = None, None
            if data['people'][0]["pose_keypoints_2d"][0] > data['people'][1]["pose_keypoints_2d"][0]:  # compare nose's x values to see which is left
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
        for i in range(0, 9):
            body_points_array[i*2] -= nose_x
            body_points_array[i*2+1] -= nose_y
    return body_points_arrays

#def normalize_body_size(body_points_arrays, ideal_neck_size=20):


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
        action_frames = []
        for curr_json in all_json_files:
            if int(frame_start) <= int(curr_json.split("_")[-2]) <= int(frame_end):
                action_frames.append(curr_json)
        # FOR EACH FRAME, DO SOMETHING
        lol = get_body_points(action_frames, focus_2ppl=ppl_focus)
        print "lol length: ", len(lol)
        #lol = centering(lol)
        #features.append("")
        labels.append(video_name[0])



