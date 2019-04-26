import json
import os
import numpy as np
import matplotlib.pyplot as plt

def normalize_coor(points):
	maxs = np.max(points,axis=0,keepdims=True)
	
	mins = np.min(points,axis=0,keepdims=True)
	return (points - mins)/(maxs-mins)
	
filepath = '../data'
files = [os.path.join(filepath,file) for file in  os.listdir(filepath)]

keypoints_pairs = [[0,15],[0,16],[15,17],[16,18],[0,1],
					[1,2],[1,5],[1,8],
					[2,3],[3,4],
					[5,6],[6,7],
					[8,9],[8,12],
					[9,10],[10,11],[11,22],[11,24],[22,23],
					[12,13],[13,14],[14,19],[14,21],[19,20]]

#keypoints_pairs = [[1,2],[1,0],[1,8],[1,5]]				
keypoints_pairs = np.array(keypoints_pairs)

for filename in files:
	with open(filename) as json_file:
		data = json.load(json_file)
		keypoints = data["people"][0]["pose_keypoints_2d"]
		
		count = 0
		keypoints_ls = []
		for idx in range(0,len(keypoints),3):
			#print(count,keypoints[idx],keypoints[idx+1],keypoints[idx+2])
			keypoints_ls.append([-keypoints[idx],-keypoints[idx+1],keypoints[idx+2]])
			
		
		keypoints_ls = np.array(keypoints_ls)
		
		keypoints_ls_normalize = normalize_coor(keypoints_ls[:,:2])
		plt.figure(figsize=(3,8))	
		for pair in keypoints_pairs:
			k, l = pair
			if(keypoints_ls[k,2] > 0 and keypoints_ls[l,2] > 0):
				print(k,l,keypoints_ls_normalize[k,:2],keypoints_ls_normalize[l,:2])
				plt.plot([keypoints_ls_normalize[k,0],keypoints_ls_normalize[l,0]],[keypoints_ls_normalize[k,1],keypoints_ls_normalize[l,1]])
		
		plt.show()