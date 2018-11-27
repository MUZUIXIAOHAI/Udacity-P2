__author__ = 'valkyrie_Z'

import os
from urllib.request import urlretrieve

import zipfile

import pickle

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd

from cnn_model import *


def show_all_kinds_images(figsize_input, X_data, y_data):
	#show the 43 kinds data
	#figsize_input - images sizes
	plt.figure(figsize=figsize_input)
	gs1 = gridspec.GridSpec(7,7)
	gs1.update(wspace=0.1, hspace=0.1)

	for i in range(43):
	    ax1 = plt.subplot(gs1[i])
	    plt.axis('on')
	    ax1.set_xticklabels([])
	    ax1.set_yticklabels([])
	    ax1.set_aspect('equal')
	    
	    index_y = np.argwhere(y_data == i)
	    ind_plot = np.random.randint(1,len(index_y))
	    plt.imshow(X_data[int(index_y[ind_plot])])
	    plt.text(3,3,str(i), color = 'k', backgroundcolor = 'c')
	    plt.axis('off')

	plt.show()

def show_datas_num(data_set):
	#show datas number and souted it
	data_i = [[i, sum(data_set == i )] for i in range(len(np.unique(data_set)))]
	data_i_sorted = sorted(data_i, key=lambda x: x[1])

	data_pd = pd.read_csv('signnames.csv')

	data_pd['Occurance'] = pd.Series(np.asarray(data_i_sorted).T[1], index=np.asarray(data_i_sorted).T[0])
	data_pd_sorted = data_pd.sort_values(['Occurance'], ascending=[0]).reset_index()
	data_pd_sorted = data_pd_sorted.drop('index', 1)

	data_pd_sorted

	plt.figure(figsize = (12, 8))
	plt.bar(range(43), height = data_pd_sorted["Occurance"])
	plt.show()

if __name__ == '__main__':
	
	url_link = "https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip"
	file_name = "traffic-signs-data.zip"
	Process_data = augment_images(url_link, file_name)

	# print("X_train_origen: " + str(X_train_origen.shape) + str(y_train_origen.shape))
	# print("X_valid_origen: " + str(X_valid_origen.shape) + str(y_valid_origen.shape))
	# print("X_test_origen: " + str(X_test_origen.shape) + str(y_test_origen.shape))

	X_train_origen, y_train_origen = Process_data.get_train_data_origen()
	X_valid_origen, y_valid_origen = Process_data.get_vaild_data_origen()
	X_test_origen, y_test_origen = Process_data.get_test_data_origen()

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

	# #show all kinds of images
	# show_all_kinds_images((8,8) , X_train_origen, y_train_origen)
	# #show datas number and souted it
	# show_datas_num(y_train_origen)

	X_train, y_train = Process_data.train_data_peocess()
	X_valid, y_valid = Process_data.vaild_data_process()
	X_test, y_test = Process_data.test_data_process()

	cnn_m = cnn_model_c()
	cnn_m.create()
	cnn_m.train(X_train, y_train, X_valid, y_valid, X_test, y_test)


