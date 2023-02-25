import os
import json
import random
import shutil
import cv2

#####remove#######
# dataset_name = 'SHAD2'
# dataset_path = '/raid/file2/code2/tsm/data/FallDet/videos_trimmed/' + dataset_name + '-frames' # 'jester-v1'
# categories = ['fall','stand','squat','sit']
# for ii, category in enumerate(categories):
# 	dir_files = os.listdir(os.path.join(dataset_path,category))
# 	for jj in range(len(dir_files)):
# 		dir_files2 = os.listdir(os.path.join(dataset_path,category,dir_files[jj]))
# 		dir_files2 = [x for x in dir_files2 if not x.endswith('.jpg')]
# 		for j in range(len(dir_files2)):
# 			if os.path.isfile(os.path.join(dataset_path, category,dir_files[jj],dir_files2[j])):
# 				os.remove(os.path.join(dataset_path, category,dir_files[jj],dir_files2[j]))
# 			elif os.path.isdir(os.path.join(dataset_path, category,dir_files[jj],dir_files2[j])):
# 				shutil.rmtree(os.path.join(dataset_path, category,dir_files[jj],dir_files2[j]))

#####copy######
# dataset_name = 'SHAD2'
# dataset_path = '/raid/file2/code2/tsm/data/FallDet/videos_trimmed/' + dataset_name + '-frames'
# dataset_path2 = '/raid/file2/code2/tsm/data/FallDet/videos_trimmed/SHAD3-frames'
# categories = ['stand']
# for ii, category in enumerate(categories):
# 	dir_files = os.listdir(os.path.join(dataset_path,category))
# 	for jj in range(len(dir_files)):
# 		dir_files2 = os.listdir(os.path.join(dataset_path,category,dir_files[jj]))
# 		if not os.path.exists(os.path.join(dataset_path2,category,dir_files[jj])):
# 			os.makedirs(os.path.join(dataset_path2,category,dir_files[jj]))
# 		for j in range(10):
# 			shutil.copytree(os.path.join(dataset_path, category,dir_files[jj]),os.path.join(dataset_path2,category,dir_files[jj],str(j)))

#######rename#######
# dataset_name = 'SHAD3'
# dataset_path = '/raid/file2/code2/tsm/data/FallDet/videos_trimmed/' + dataset_name + '-frames'
# categories = ['fall','stand','squat','sit']
# for ii, category in enumerate(categories):
# 	dir_files = os.listdir(os.path.join(dataset_path,category))
# 	for jj in range(len(dir_files)):
# 		dir_files2 = os.listdir(os.path.join(dataset_path,category,dir_files[jj]))
# 		for j in range(len(dir_files2)):
# 			dir_files3 = os.listdir(os.path.join(dataset_path,category,dir_files[jj],dir_files2[j]))
# 			for k in range(len(dir_files3)):
# 				newname = os.path.join(dataset_path, category, dir_files[jj],dir_files2[j],str(k+1).zfill(4))+ '.jpg'
# 				os.rename(os.path.join(dataset_path, category, dir_files[jj],dir_files2[j],dir_files3[k]),newname)


#####crop######
dataset_name = 'SHAD3'
dataset_path = '/raid/file2/code2/tsm/data/FallDet/videos_trimmed/' + dataset_name + '-frames'
dataset_path2 = '/raid/file2/code2/tsm/data/FallDet/videos_trimmed/SHAD3-crop-frames'
categories = ['fall','stand','squat','sit']
crops_list = ['crop1','crop2','crop3','crop4','crop5']
# for ii, category in enumerate(categories):
# 	dir_files = os.listdir(os.path.join(dataset_path,category))
# 	for jj in range(len(dir_files)):
# 		dir_files2 = os.listdir(os.path.join(dataset_path,category,dir_files[jj]))
# 		for i in range(len(dir_files2)):
# 			for j, crop_list in enumerate(crops_list):
# 				shutil.copytree(os.path.join(dataset_path, category,dir_files[jj],dir_files2[i]),os.path.join(dataset_path2,category,dir_files[jj],crop_list,dir_files2[i]))
for ii, category in enumerate(categories):
	dir_files = os.listdir(os.path.join(dataset_path2,category))
	# print(dir_files[0][11:14])
	for jj, dir_file in enumerate(dir_files):
		if dir_file[11:14] == '135':
			scales = [0.875,0.75,0.66,0.55]
			offset_H = 150
			offset_W = 80

		elif dir_file[11:14] == '172':
			scales = [0.875,0.75,0.66,0.55]
			offset_H = -50
			offset_W = -50

		elif dir_file[11:14] == '173':
			scales = [0.875,0.75,0.66,0.55]
			offset_H = 100
			offset_W = 50

		elif dir_file[11:14] == '175':
			scales = [0.875,0.75,0.66,0.55]
			offset_H = 0
			offset_W = 50
		elif dir_file[11:14] == '179':
			scales = [0.875,0.75,0.66,0.55]
			if category == 'stand':
				offset_H = 300
				scales = [0.975,0.875,0.75,0.66]
			else:
				offset_H = -60
			offset_W = 300

		elif dir_file[11:14] == '182':
			scales = [0.975,0.875,0.75,0.66]
			offset_H = 0
			offset_W = -350
			if category == 'sit':
				offset_H = 100
				offset_W = -450
			elif category == 'squat':
				offset_W = -100
		elif dir_file[11:14] == '180':
			scales = [0.875,0.75,0.66,0.55]
			offset_H = 250
			offset_W = 200
			if category == 'sit':
				offset_H = 250
				offset_W = 200
			elif category == 'squat':
				offset_H = 250
				offset_W = 200
				scales = [0.975,0.875,0.75,0.66]
		elif dir_file[11:13] == '94':
			scales = [0.875,0.75,0.66,0.55]
			offset_H = 50
			offset_W = -100
		elif dir_file[11:14] == '136':
			scales = [0.875,0.75,0.66,0.55]
			offset_H = 200
			offset_W = 100
		elif dir_file[11:14] == '170':
			scales = [0.975,0.875,0.75,0.66]
			offset_H = 100
			offset_W = -250
		elif dir_file[11:14] == '171':
			scales = [0.975,0.875,0.75,0.66]
			offset_H = 50
			offset_W = 350
		elif dir_file[11:14] == '178':
			scales = [0.975,0.875,0.75,0.66]
			offset_H = 250
			offset_W = 150
		else:
			print('missing files:',dir_file)
		for k , scale in enumerate(scales):
			dir_files2 = os.listdir(os.path.join(dataset_path2,category,dir_file,crops_list[k]))
			for i, dir_file2 in enumerate(dir_files2):
				img_files = os.listdir(os.path.join(dataset_path2,category,dir_file,crops_list[k],dir_file2))
				for j , img_file in enumerate(img_files):
					img = cv2.imread(os.path.join(dataset_path2,category,dir_file,crops_list[k],dir_file2,img_file))
					H, W, _ = img.shape
					basesize = min(H,W)
					crop_size = int(basesize*scale)
					h1 = max(int(0.5*H+offset_H-0.5*crop_size),0)
					h2 = min(int(0.5*H+offset_H+0.5*crop_size),H)
					w1 = max(int(0.5*W+offset_W-0.5*crop_size),0)
					w2 = min(int(0.5*W+offset_W+0.5*crop_size),W)
					img_crop = img[h1:h2,w1:w2]
					cv2.imwrite(os.path.join(dataset_path2,category,dir_file,crops_list[k],dir_file2,img_file),img_crop)

		print('{}-{}/{}'.format(category,jj,len(dir_files)))

