
import os
import sys
import math
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
from scipy.signal import butter, lfilter
from pathlib import Path

  
from pywt import Wavelet
from math import floor, ceil
from numpy import concatenate, flipud, zeros, convolve, array

def padding_symmetric(signal, size=8):
	'''
	Applies a symmetric padding of the specified size to the input signal.
	Parameters
	----------
	signal : ndarray
		The signal to be padded.
	size : int, optional
		The size of the padding which corresponds to the size of the filter. The default is 8.
	Returns
	-------
	padded_signal : ndarray
		Padded signal.
	'''
	
	padded_signal = concatenate([flipud(signal[:size]), signal, flipud(signal[-size:])])
	return padded_signal


def restore_signal(signal, reconstruction_filter, real_len):
	'''
	Restores the signal to its original size using the reconstruction filter.
	Parameters
	----------
	signal : ndarray
		The signal to be restored.
	reconstruction_filter : list
		The reconstruction filter to be used for restoring the signal.
	real_len : int
		Real length of the signal.
	Returns
	-------
	restored_signal : ndarray
		Restored signal of the specified length.
	'''
	restored_signal = zeros(2 * len(signal) + 1)
	for i in range(len(signal)):
		restored_signal[i*2+1] = signal[i]
	restored_signal = convolve(restored_signal, reconstruction_filter)
	restored_len = len(restored_signal)
	exceed_len = (restored_len - real_len) / 2
	restored_signal = restored_signal[int(floor(exceed_len)):(restored_len - int(ceil(exceed_len)))]
	return restored_signal

def DWTfn(signal, level=3, mother_wavelet='db4'):
	'''
	Applies a Discrete Wavelet Transform to the signal.
	Parameters
	----------
	signal : ndarray
		The signal on which the DWT will be applied.
	level : int, optional
		The decomposition levels for the DWT. The default is 3.
	mother_wavelet : str, optional
		The mother wavelet that it is going to be used in the DWT. The default is "db4".
	Returns
	-------
	restored_approx_coeff : list
		Restored approximations coefficients.
	restored_detail_coeff : list
		Restored detail coefficients.
	'''
	if type(signal).__name__ != "ndarray" and type(signal) != list:
		raise TypeError(f"'signal' must be 'ndarray', received: '{type(signal).__name__}'")
	if type(signal) == list:
		signal = array(signal)
	if "float" not in signal.dtype.name and "int" not in signal.dtype.name:
		raise TypeError(f"All elements of 'signal' must be numbers")
		   
	if type(level) != int:
		raise TypeError(f"'level' must be 'int', received: '{type(level).__name__}'")
	if level < 1:
		raise TypeError(f"'level' must be greater than 0, received: {level}")
		
	if mother_wavelet not in ['haar', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8', 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8', 'dmey', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'shan', 'fbsp', 'cmor']:
		raise TypeError(f"Invalid 'mother_wavelet' must be 'haar', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8', 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8', 'dmey', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'shan', 'fbsp', or 'cmor', received: '{mother_wavelet}'")
		
	original_len = len(signal)
	approx_coeff = []
	detail_coeff = []
	wavelet = pywt.Wavelet(mother_wavelet)
	low_filter = wavelet.dec_lo
	high_filter = wavelet.dec_hi
	filter_size = len(low_filter)
	try:
		for _ in range(level):
			padded_signal = padding_symmetric(signal, filter_size)
			low_pass_filtered_signal = convolve(padded_signal, low_filter)[filter_size:(2*filter_size)+len(signal)-1] 
			low_pass_filtered_signal = low_pass_filtered_signal[1:len(low_pass_filtered_signal):2]
			high_pass_filtered_signal = convolve(padded_signal, high_filter)[filter_size:filter_size+len(signal)+filter_size-1]
			high_pass_filtered_signal = high_pass_filtered_signal[1:len(high_pass_filtered_signal):2]
			approx_coeff.append(low_pass_filtered_signal)
			detail_coeff.append(high_pass_filtered_signal)
			signal = low_pass_filtered_signal
	except:
		raise
	low_reconstruction_filter = wavelet.rec_lo
	high_reconstruction_filter = wavelet.rec_hi
	real_lengths = []
	for i in range(level-2,-1,-1):
		real_lengths.append(len(approx_coeff[i]))
	real_lengths.append(original_len)
	restored_approx_coeff = []
	for i in range(level):
		restored_signal = restore_signal(approx_coeff[i], low_reconstruction_filter, real_lengths[level-1-i])
		for j in range(i):
			restored_signal = restore_signal(restored_signal, low_reconstruction_filter, real_lengths[level-i+j])
		restored_approx_coeff.append(restored_signal)
	restored_detail_coeff = []
	for i in range(level):
		restored_signal = restore_signal(detail_coeff[i], high_reconstruction_filter, real_lengths[level-1-i])
		for j in range(i):
			restored_signal = restore_signal(restored_signal, high_reconstruction_filter, real_lengths[level-i+j])
		restored_detail_coeff.append(restored_signal)
	return restored_approx_coeff, restored_detail_coeff 

def read_file(file):
	data = sio.loadmat(file)
	data = data['data']
	# print(data.shape)
	return data

def entropy_fn(signal):
	entropy_val = 0
	for i in signal:
		entropy_val += (i**2)*(np.log2(i**2))     
	return entropy_val

def energy_fn(signal):
	return np.sum(np.array(signal)**2)

def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y

def compute_DE(signal):
	variance = np.var(signal,ddof=1)
	return math.log(2*math.pi*math.e*variance)/2
		
import pywt
def dwt_fn(signal, feature_method):
	restored_approx_coeff,restored_detail_coeff = DWTfn(signal, 4, 'db4') 
	d4, d3, d2, d1 = restored_detail_coeff 
	
	if feature_method == "entropy":
		theta = entropy_fn(d4)	
		alpha = entropy_fn(d3)
		beta = entropy_fn(d2)
		gamma = entropy_fn(d1)
	elif feature_method == "energy":
		theta = energy_fn(d4)	
		alpha = energy_fn(d3)
		beta = energy_fn(d2)
		gamma = energy_fn(d1)
	
	return (theta, alpha, beta, gamma)

def dwt_fn_baseline(signal, feature_method, length):

	if length == 3:
		splits = [[0,384]]
	elif length == 1:
		splits = [[0,128],[128,256],[256,384]]

	theta = []
	alpha = []
	beta = []
	gamma = []

	for sec_split in splits:
		signal_temp = signal[sec_split[0]:sec_split[1]]
		restored_approx_coeff,restored_detail_coeff = DWTfn(signal_temp, 4, 'db4') 
		d4, d3, d2, d1 = restored_detail_coeff 

		bands = [d4, d3, d2, d1]
		band_index = 0
		for band in bands:
			if feature_method == "entropy": feature = entropy_fn(band)
			elif feature_method == "energy": feature = energy_fn(band)
			else: raise Exception("Feature method must either be entropy or energy.")

			if band_index == 0: theta.append(feature)
			elif band_index == 1: alpha.append(feature)
			elif band_index == 2: beta.append(feature)
			elif band_index == 3: gamma.append(feature)
			band_index+=1	

	return (np.mean(theta), np.mean(alpha), np.mean(beta), np.mean(gamma))


def de_fn(signal):
	theta = butter_bandpass_filter(signal, 4, 8, 128, order=3)
	alpha = butter_bandpass_filter(signal, 8,14, 128, order=3)
	beta = butter_bandpass_filter(signal, 14,31, 128, order=3)
	gamma = butter_bandpass_filter(signal,31,45, 128, order=3)	
	
	theta = compute_DE(theta)
	alpha = compute_DE(alpha)
	beta = compute_DE(beta)
	gamma = compute_DE(gamma)
	
	return (theta, alpha, beta, gamma)

def de_fn_baseline(signal, length):
	if length == 3:
		splits = [[0,384]]
	elif length == 1:
		splits = [[0,128],[128,256],[256,384]]

	theta = []
	alpha = []
	beta = []
	gamma = []

	base_theta = butter_bandpass_filter(signal, 4, 8, 128, order=3)
	base_alpha = butter_bandpass_filter(signal, 8,14, 128, order=3)
	base_beta = butter_bandpass_filter(signal, 14,31, 128, order=3)
	base_gamma = butter_bandpass_filter(signal,31,45, 128, order=3)
	bands = [base_theta, base_alpha, base_beta, base_gamma]

	for sec_split in splits:
		band_index = 0
		for band in bands:
			feature = compute_DE(band[sec_split[0]:sec_split[1]])

			if band_index == 0: theta.append(feature)
			elif band_index == 1: alpha.append(feature)
			elif band_index == 2: beta.append(feature)
			elif band_index == 3: gamma.append(feature)
			band_index+=1	

	return (np.mean(theta), np.mean(alpha), np.mean(beta), np.mean(gamma))


#----------------------------------------------------------------------------------------------------------------------------------


def decompose(file, channel, length, feature, custom=False):
	
	if channel == 5: 
		channels = [1,7,15,17,25]
		if custom:
			channels = [0, 1, 2, 3, 4]
	elif channel == 32: channels = list(np.arange(0,32))
	else: raise Exception("Channels must be either 5 or 32.")

	# trial*channel*sample
	start_index = 384 #3s pre-trial signals
	data = read_file(file)
	shape = data.shape
	frequency = 128

	if length == 3: decomposed_de = np.empty([0,4,20])
	elif length == 1: decomposed_de = np.empty([0,4,60])
	else: raise Exception("Length must be either 1 or 3 sec.")

	if channel == 5: base_DE = np.empty([0,20])
	elif channel == 32: base_DE = np.empty([0,128])	
	
	for trial in range(len(data)):
		temp_base_DE = np.empty([0])
		temp_base_theta_DE = np.empty([0])
		temp_base_alpha_DE = np.empty([0])
		temp_base_beta_DE = np.empty([0])
		temp_base_gamma_DE = np.empty([0])
		
		
		if length == 3: temp_de = np.empty([0,20])
		elif length == 1: temp_de = np.empty([0,60])
		# if channel == 5: temp_de = np.empty([0,20])
		# elif channel == 32: temp_de = np.empty([0,60])

		for chan in channels:			
			trial_signal = data[trial,chan,384:]
			base_signal = data[trial,chan,:384]
			#****************compute baseline DWT****************

			if feature == "DE":
				base_theta_DE, base_alpha_DE, base_beta_DE, base_gamma_DE = de_fn_baseline(base_signal, length)
			else:
				base_theta_DE, base_alpha_DE, base_beta_DE, base_gamma_DE = dwt_fn_baseline(base_signal, feature, length)
			

			temp_base_theta_DE = np.append(temp_base_theta_DE,base_theta_DE)
			temp_base_gamma_DE = np.append(temp_base_gamma_DE,base_gamma_DE)
			temp_base_beta_DE = np.append(temp_base_beta_DE,base_beta_DE)
			temp_base_alpha_DE = np.append(temp_base_alpha_DE,base_alpha_DE)

			DE_theta = np.zeros(shape=[0],dtype = float)
			DE_alpha = np.zeros(shape=[0],dtype = float)
			DE_beta =  np.zeros(shape=[0],dtype = float)
			DE_gamma = np.zeros(shape=[0],dtype = float)

			for index in range(60):
				if length == 1:
					if feature == "DE": theta_dwt, alpha_dwt, beta_dwt, gamma_dwt = de_fn(trial_signal[index*frequency:(index+1)*frequency])
					else: theta_dwt, alpha_dwt, beta_dwt, gamma_dwt = dwt_fn(trial_signal[index*frequency:(index+1)*frequency], feature)
					
					DE_theta =np.append(DE_theta,theta_dwt)
					DE_alpha =np.append(DE_alpha,alpha_dwt)
					DE_beta =np.append(DE_beta,beta_dwt)
					DE_gamma =np.append(DE_gamma,gamma_dwt)			
				elif length == 3:
					if index == 0 or index % 3 == 0: 
						if feature == "DE": theta_dwt, alpha_dwt, beta_dwt, gamma_dwt = de_fn(trial_signal[index*frequency:(index+1)*frequency])
						else: theta_dwt, alpha_dwt, beta_dwt, gamma_dwt = dwt_fn(trial_signal[index*frequency:(index+3)*frequency], feature)						

						DE_theta =np.append(DE_theta,theta_dwt)
						DE_alpha =np.append(DE_alpha,alpha_dwt)
						DE_beta =np.append(DE_beta,beta_dwt)
						DE_gamma =np.append(DE_gamma,gamma_dwt)

			temp_de = np.vstack([temp_de,DE_theta])
			temp_de = np.vstack([temp_de,DE_alpha])
			temp_de = np.vstack([temp_de,DE_beta])
			temp_de = np.vstack([temp_de,DE_gamma])

		if length == 3: temp_trial_de = temp_de.reshape(-1,4,20)
		if length == 1: temp_trial_de = temp_de.reshape(-1,4,60)
		decomposed_de = np.vstack([decomposed_de,temp_trial_de])

		temp_base_DE = np.append(temp_base_theta_DE,temp_base_alpha_DE)
		temp_base_DE = np.append(temp_base_DE,temp_base_beta_DE)
		temp_base_DE = np.append(temp_base_DE,temp_base_gamma_DE)
		base_DE = np.vstack([base_DE,temp_base_DE])
	
	decomposed_de = decomposed_de.reshape(-1,channel,4,int(60/length)).transpose([0,3,2,1]).reshape(-1,4,channel).reshape(-1,channel*4)

	print("base_DE shape:",base_DE.shape)
	print("trial_DE shape:",decomposed_de.shape)
	return base_DE,decomposed_de

def get_labels(file, length):
	#0 valence, 1 arousal, 2 dominance, 3 liking
	valence_labels = sio.loadmat(file)["labels"][:,0]>5	# valence labels
	arousal_labels = sio.loadmat(file)["labels"][:,1]>5	# arousal labels
	final_valence_labels = np.empty([0])
	final_arousal_labels = np.empty([0])
	for i in range(len(valence_labels)):
		for j in range(0,60):
			if length == 1:
				final_valence_labels = np.append(final_valence_labels,valence_labels[i])
				final_arousal_labels = np.append(final_arousal_labels,arousal_labels[i])
			elif length == 3:
				if j == 0 or j % 3 == 0:
					final_valence_labels = np.append(final_valence_labels,valence_labels[i])
					final_arousal_labels = np.append(final_arousal_labels,arousal_labels[i])
	print("labels:",final_arousal_labels.shape)
	return final_arousal_labels,final_valence_labels

def wgn(x, snr):
	snr = 10**(snr/10.0)
	xpower = np.sum(x**2)/len(x)
	npower = xpower / snr
	return np.random.randn(len(x)) * np.sqrt(npower)

def feature_normalize(data):
	mean = data[data.nonzero()].mean()
	sigma = data[data. nonzero ()].std()
	data_normalized = data
	data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
	return data_normalized


if __name__ == '__main__':	

	channel_dir = sys.argv[1]
	length_dir = sys.argv[2]
	feature_type = sys.argv[3] #entropy, energy DE

	custom = False
	dataset_dir = "../data_preprocessed_matlab/"
	if len(sys.argv) == 5:		
		custom = True
		dataset_dir = "../DEAP_5chan_custom_preproc/"

	if custom and int(channel_dir) == 32:
		raise Exception("Custom dataset is only available for 5 channels.")

	if custom: result_dir = str(os.getcwd())+f"/1D_dataset_{feature_type}_custom/{channel_dir}chan/{length_dir}sec/"
	else: result_dir = str(os.getcwd())+f"/1D_dataset_{feature_type}/{channel_dir}chan/{length_dir}sec/"

	if os.path.isdir(result_dir)==False:
		os.makedirs(result_dir)

	for file in os.listdir(dataset_dir):
		print("processing: ",file,"......")		
		file_path = os.path.join(dataset_dir,file)
		if os.path.exists(result_dir+"DE_"+file):
			print("File exists. Skipping.")
			continue
		base_DE,trial_DE = decompose(file_path, int(channel_dir), int(length_dir), feature_type, custom)
		arousal_labels,valence_labels = get_labels(file_path, int(length_dir))
		sio.savemat(result_dir+"DE_"+file,{"base_data":base_DE,"data":trial_DE,"valence_labels":valence_labels,"arousal_labels":arousal_labels})

#python get_1D_data.py 5 1 DE
#python get_1D_data.py 5 1 entropy yes