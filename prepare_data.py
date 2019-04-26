import os

data_path = './data/'

lines_video_list = open(data_path + "video_list.csv", "r").readlines()
videos = [line_video_list.rstrip('\n').split(",")[0] for line_video_list in lines_video_list]

features = []
labels = []

for line_video_list in lines_video_list:  # for each video
    # basic info
    video_name = line_video_list.split(",")[0]
    ppl_focus = line_video_list.split(",")[1]

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
        action_frames = [json for json in all_json_files if frame_start <= int(json.split("_")[-2]) <= frame_end]
        # FOR EACH FRAME, DO SOMETHING
        #features.append("")  call extract features
        labels.append(video_name[0])

def extract_features(filenames, num_of_ppl=1, focus_2ppl="none"):


    return -1