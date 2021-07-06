
import time
from sklearn import preprocessing
from scipy.signal import butter, lfilter
import scipy.io as sio
import numpy as np
import os
import math
import sys

def read_file(file):
	file = sio.loadmat(file)
	trial_data = file['data']
	base_data = file["base_data"]
	return trial_data,base_data,file["arousal_labels"],file["valence_labels"]

def get_vector_deviation(vector1,vector2):
	return vector1-vector2

def get_dataset_deviation(trial_data,base_data, channel, length):
	if channel == 32:
		new_dataset = np.empty([0,128])
	elif channel == 5:
		new_dataset = np.empty([0,20])

	if length == 3: length_trial = 800
	elif length == 1: length_trial = 2400 	
	for i in range(0,length_trial):
		if length == 3: base_index = i//20
		if length == 1: base_index = i//60
		base_index = 39 if base_index == 40 else base_index
		if channel == 32: new_record = get_vector_deviation(trial_data[i],base_data[base_index]).reshape(1,128)
		elif channel == 5: new_record = get_vector_deviation(trial_data[i],base_data[base_index]).reshape(1,20)

		new_dataset = np.vstack([new_dataset,new_record])
		
	return new_dataset

#[1,7,15,17,25]
#0: 1 AF3
#1: 7 T7
#2: 15 Pz
#3: 17 AF4
#4: 25 T8

def data_1Dto2D(data, Y, X):
	data_2D = np.zeros([Y, X])

	if X == 9:
		data_2D[0] = (0,        0,          0,          data[0],    0,          data[16],   0,          0,          0       )
		data_2D[1] = (0,        0,          0,          data[1],    0,          data[17],   0,          0,          0       )
		data_2D[2] = (data[3],  0,          data[2],    0,          data[18],   0,          data[19],   0,          data[20])
		data_2D[3] = (0,        data[4],    0,          data[5],    0,          data[22],   0,          data[21],   0       )
		data_2D[4] = (data[7],  0,          data[6],    0,          data[23],   0,          data[24],   0,          data[25])
		data_2D[5] = (0,        data[8],    0,          data[9],    0,          data[27],   0,          data[26],   0       )
		data_2D[6] = (data[11], 0,          data[10],   0,          data[15],   0,          data[28],   0,          data[29])
		data_2D[7] = (0,        0,          0,          data[12],   0,          data[30],   0,          0,          0       )
		data_2D[8] = (0,        0,          0,          data[13],   data[14],   data[31],   0,          0,          0       )
	
	elif X == 5:
		data_2D[0] = (0,        data[0],    0,          data[3],    0       )
		data_2D[1] = (0,        0,    		0,          0,		    0       )
		data_2D[2] = (data[1],  0,          0,		    0,          data[4] )	
		data_2D[3] = (0,   		0,          data[2],    0,          0		)	
		data_2D[4] = (0,   		0,          0,		    0,          0		)	

	return data_2D

def pre_process(path,y_n, channel, length):
	# feature vector dimension of each band 
	if channel == 32: data_3D = np.empty([0,9,9])
	elif channel == 5: data_3D = np.empty([0,5,5])

	trial_data,base_data,arousal_labels,valence_labels = read_file(path)

	if y_n=="yes":
		data = get_dataset_deviation(trial_data,base_data, channel, length)
		data = preprocessing.scale(data,axis=1, with_mean=True,with_std=True,copy=True)
	else:
		data = preprocessing.scale(trial_data,axis=1, with_mean=True,with_std=True,copy=True)
		
	for vector in data:
		for band in range(0,4):
			if channel == 32:
				data_2D_temp = data_1Dto2D(vector[band*32:(band+1)*32],9,9)
				data_2D_temp = data_2D_temp.reshape(1,9,9)
			elif channel == 5:
				data_2D_temp = data_1Dto2D(vector[band*5:(band+1)*5],5,5)
				data_2D_temp = data_2D_temp.reshape(1,5,5)						
			data_3D = np.vstack([data_3D,data_2D_temp])

	if channel == 32: data_3D = data_3D.reshape(-1,4,9,9)
	elif channel == 5: data_3D = data_3D.reshape(-1,4,5,5)

	print("final data shape:",data_3D.shape)
	return data_3D,arousal_labels,valence_labels

if __name__ == '__main__':
	cwd = str(os.getcwd())
	use_baseline = sys.argv[1]
	channel_dir = sys.argv[2]
	length_dir = sys.argv[3]
	feature_type = sys.argv[4] # entropy, energy, DE

	custom = False
	dataset_dir = cwd+f"/1D_dataset_{feature_type}/{channel_dir}chan/{length_dir}sec/"
	if len(sys.argv) == 6:		
		custom = True
		dataset_dir = cwd+f"/1D_dataset_{feature_type}_custom/{channel_dir}chan/{length_dir}sec/"

	if custom and int(channel_dir) == 32:
		raise Exception("Custom dataset is only available for 5 channels.")

	if custom: custom_dir = "_custom"
	else: custom_dir = ""
	
	if use_baseline=="yes":
		result_dir = cwd+f"/3D_dataset_{feature_type}{custom_dir}/{channel_dir}chan/{length_dir}sec/with_base/"
		if os.path.isdir(result_dir)==False:
			os.makedirs(result_dir)
	else:
		result_dir = cwd+f"/3D_dataset_{feature_type}{custom_dir}/{channel_dir}chan/{length_dir}sec/without_base/"
		if os.path.isdir(result_dir)==False:
			os.makedirs(result_dir)

	for file in os.listdir(dataset_dir):
		print("processing: ",file,"......")
		file_path = os.path.join(dataset_dir,file)		
		if os.path.exists(result_dir+file):
			print("File exists. Skipping.")
			continue
		data,arousal_labels,valence_labels = pre_process(file_path,use_baseline, int(channel_dir), int(length_dir))
		print("final shape:",data.shape)
		print(result_dir)
		sio.savemat(result_dir+file,{"data":data,"valence_labels":valence_labels,"arousal_labels":arousal_labels})
		#break

#python 1D_to_3D.py yes 5 1 DE