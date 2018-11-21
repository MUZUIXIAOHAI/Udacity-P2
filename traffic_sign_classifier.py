__author__ = 'valkyrie_Z'

import os
from urllib.request import urlretrieve

import zipfile

import pickle

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def download(url, file):
	'''
	Download file form <url>
	:param url:URL to file form internet
	:param file:Local file path
	'''
	if not os.path.isfile(file):
		print('Downloading ' + file + '...')
		urlretrieve(url, file)
		print('Download Finished')


def un_zip(file_name):
	"""
	unzip the file 
	"""
	zip_file = zipfile.ZipFile(file_name)
	file_name_front = os.path.splitext(file_name)
	print("Unziping...")

	if os.path.isdir(file_name_front[0] + "_files"):
		pass
	else:
		os.mkdir(file_name_front[0] + "_files")
	for names in zip_file.namelist():
		zip_file.extract(names,file_name_front[0] + "_files/")
	print("Unzip finish")
	zip_file.close()



if __name__ == '__main__':
	#if file did not existed, download it
	if os.path.exists("./traffic-signs-data.zip"):
		print("pass, file has been existed")
	else:
		download('https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip',\
		 'traffic-signs-data.zip')

	#if file did not unzip, unzip it
	if os.path.exists("./traffic-signs-data_files/"):
		print("pass, file has been unzip")
	else:
		un_zip('traffic-signs-data.zip')


	#load the tainning data
	training_file = 'traffic-signs-data_files/train.p'
	validation_file= 'traffic-signs-data_files/valid.p'
	testing_file = 'traffic-signs-data_files/test.p'

	with open(training_file, mode='rb') as f:
		train = pickle.load(f)
	with open(validation_file, mode='rb') as f:
		valid = pickle.load(f)
	with open(testing_file, mode='rb') as f:
		test = pickle.load(f)
	    
	X_train_origen, y_train_origen = train['features'], train['labels']
	X_valid_origen, y_valid_origen = valid['features'], valid['labels']
	X_test_origen, y_test_origen = test['features'], test['labels']

	# print("X_train_origen: " + str(X_train_origen.shape) + str(y_train_origen.shape))
	# print("X_valid_origen: " + str(X_valid_origen.shape) + str(y_valid_origen.shape))
	# print("X_test_origen: " + str(X_test_origen.shape) + str(y_test_origen.shape))

	# Number of training examples
	n_train = X_train_origen.shape[0]

	# Number of validation examples
	n_validation = X_valid_origen.shape[0]

	# Number of testing examples.
	n_test = X_test_origen.shape[0]

	# What's the shape of an traffic sign image?
	image_shape = X_train_origen[0].shape

	# How many unique classes/labels there are in the dataset.
	n_classes = len(np.unique(y_train_origen))

	print("Number of training examples =", n_train)
	print("Number of testing examples =", n_test)
	print("Number of valid examples = ", n_validation)
	print("Image data shape =", image_shape)
	print("Number of classes =", n_classes)


	#show the 43 kinds data
	plt.figure(figsize=(16,18))
	gs1 = gridspec.GridSpec(9,5)
	gs1.update(wspace=0.01, hspace=0.02)

	for i in range(43):
	    ax1 = plt.subplot(gs1[i])
	    plt.axis('on')
	    ax1.set_xticklabels([])
	    ax1.set_yticklabels([])
	    ax1.set_aspect('equal')
	    
	    index_y = np.argwhere(y_train_origen == i)
	    ind_plot = np.random.randint(1,len(index_y))
	    plt.imshow(X_train_origen[int(index_y[ind_plot])])
	    plt.text(2,4,str(i), color = 'k', backgroundcolor = 'c')
	    plt.axis('off')

	plt.show()




