import cv2
import numpy as np

import os
from urllib.request import urlretrieve

import zipfile

import pickle

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd

from tensorflow.contrib.layers import flatten

import tensorflow as tf

from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

class augment_images(object):
	"""docstring for augment_images"""
	def __init__(self, url, file_name):
		"""
		download & unzip & load the data
		url - the link where the data download from
		file_name - downloaded file name
		"""
		self.ang_rot = 10
		self.trans_rot = 2
		self.shear_rot = 2

		#if file did not existed, download it
		if os.path.exists("./" + file_name):
			print("pass, file has been existed")
		else:
			self.download(url, file_name)

		#if file did not unzip, unzip it
		self.file_name_front = os.path.splitext(file_name)
		self.file_unzip_path = "./" + self.file_name_front[0] + "_files/"
		if os.path.exists(self.file_unzip_path):
			print("pass, file has been unzip")
		else:
			un_zip(file_name)

		#load the tainning data
		self.training_file = self.file_unzip_path + "train.p"
		self.validation_file= self.file_unzip_path + "valid.p"
		self.testing_file = self.file_unzip_path + "test.p"

		with open(self.training_file, mode='rb') as f:
			self.train = pickle.load(f)
		with open(self.validation_file, mode='rb') as f:
			self.valid = pickle.load(f)
		with open(self.testing_file, mode='rb') as f:
			self.test = pickle.load(f)

		self.X_train_origen, self.y_train_origen = self.train['features'], self.train['labels']
		self.X_valid_origen, self.y_valid_origen = self.valid['features'], self.valid['labels']
		self.X_test_origen, self.y_test_origen = self.test['features'], self.test['labels']

	def get_train_data_origen(self):
		return self.X_train_origen, self.y_train_origen

	def get_vaild_data_origen(self):
		return self.X_valid_origen, self.y_valid_origen

	def get_test_data_origen(self):
		return self.X_test_origen, self.y_test_origen

	def download(self, url, file):
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

	def augment_brightness_camera_images(self, image):
		#Random enhancement of image
		image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
		image1 = np.array(image1, dtype=np.float64)
		random_bright = .5 + np.random.uniform()
		image1[:,:,2] = image1[:,:,2]*random_bright
		image1[:,:,2][image1[:,:,2]>255] = 255
		image1 = np.array(image1, dtype=np.uint8)
		image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
		return image1

	def pre_process_image(self, image):
		#Normalization of image data

		#image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
		#image = image[:,:,0]
		image[:,:,0] = cv2.equalizeHist(image[:,:,0])
		image[:,:,1] = cv2.equalizeHist(image[:,:,1])
		image[:,:,2] = cv2.equalizeHist(image[:,:,2])
		image = image/255.-.5
		#image = cv2.resize(image, (img_resize,img_resize),interpolation = cv2.INTER_CUBIC)
		return image

	def transform_image(self, image, ang_range, shear_range, trans_range):
		#Transform enhancement of images
		#ang_range - picture rotation angle
		#shear_range - picture clipping range
		#trans_range - 
		rows, cols, ch = image.shape
		#Rotation
		ang_rot = np.random.uniform(ang_range) - ang_range/2
		Rot_M = cv2.getRotationMatrix2D((cols/2, rows/2), ang_rot, 1)

		#Translations
		tr_x = trans_range * np.random.uniform() - trans_range/2
		tr_y = trans_range * np.random.uniform() - trans_range/2
		Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

		#shear
		pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

		pt1 = 5 + shear_range * np.random.uniform() - shear_range/2
		pt2 = 20 + shear_range * np.random.uniform() - shear_range/2

		pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

		Shear_M = cv2.getAffineTransform(pts1, pts2)

		image = cv2.warpAffine(image, Rot_M, (cols, rows))
		image = cv2.warpAffine(image, Trans_M, (cols, rows))
		image = cv2.warpAffine(image, Shear_M, (cols, rows))

		image = self.augment_brightness_camera_images(image)

		image = np.array(image)

		image = self.pre_process_image(image)

		return image

	def get_extra_data(self, X_train, y_train, n_each = 5, ang_range = 10, shear_range = 2, trans_range = 2, randomize_Var = 1):
		#Enhance image, increase image data
		#n_each - the number of pictures added (default is 5)
		#ang_range - picture rotation angle
		#shear_range - picture clipping range
		#trans_range - 
		#randomize_Var - option of disrupting data (default is 1)
		X_arr = []
		Y_arr = []
		for i in range(len(X_train)):
			for i_n in range(n_each):
				img_trf = self.transform_image(X_train[i], ang_range, shear_range, trans_range)
				X_arr.append(img_trf)
				Y_arr.append(y_train[i])

		X_arr = np.array(X_arr, dtype = np.float32())
		Y_arr = np.array(Y_arr, dtype = np.float32())

		if (randomize_Var == 1 ):
			len_arr = np.arange(len(X_arr))
			np.random.shuffle(len_arr)
			X_arr[len_arr] = X_arr
			Y_arr[len_arr] = Y_arr

		return X_arr, Y_arr


	def train_data_peocess(self):
		#train_data process
		return self.get_extra_data(self.X_train_origen, self.y_train_origen)

	def vaild_data_process(self):
		#vaild_data process
		X_data = np.array([self.pre_process_image(self.X_valid_origen[i]) for i in range(len(self.X_valid_origen))], dtype = np.float32)
		return X_data, self.y_valid_origen

	def test_data_process(self):
		#test_data process
		X_data = np.array([self.pre_process_image(self.X_test_origen[i]) for i in range(len(self.X_test_origen))], dtype = np.float32)
		return X_data, self.y_test_origen


class cnn_model_c(object):
	"""docstring for cnn_model_c"""
	def __init__(self):
		self.EPOCHS = 15
		self.BATCH_SIZE = 128

		#define the input
		self.x = tf.placeholder(tf.float32, (None, 32, 32, 3))
		self.y = tf.placeholder(tf.int32, (None))
		self.keep_prob = tf.placeholder(tf.float32)
		self.one_hot_y = tf.one_hot(self.y, 43)

	def create(self):
		"""
		create the model 
		"""
		#create a training pipeline
		rate = 0.001
		logits,regularizers = self.LeNet(self.x)
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = self.one_hot_y, logits = logits)
		loss_operation = tf.reduce_mean(cross_entropy)+1e-5*regularizers
		optimizer = tf.train.AdamOptimizer(learning_rate = rate)
		self.train_operation = optimizer.minimize(loss_operation)
		#Evaluate how well the loss and accuracy of the model for a given dataset
		self.correct_perdiction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.one_hot_y, 1))
		self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_perdiction, tf.float32))

	def evaluate(self, X_data, y_data):
		"""
		evaluate the model
		"""
		num_examples = len(X_data)
		total_accuracy = 0
		sess = tf.get_default_session()
		for offset in range(0, num_examples, self.BATCH_SIZE):
			batch_x, batch_y = X_data[offset:offset+self.BATCH_SIZE], y_data[offset:offset+self.BATCH_SIZE]
			accuracy = sess.run(self.accuracy_operation, feed_dict = {self.x:batch_x, self.y:batch_y, self.keep_prob:1})
			total_accuracy += (accuracy*len(batch_x))
		return total_accuracy/num_examples

	def train(self, X_train, y_train, X_valid, y_valid, X_test, y_test):
		total_iterations = 0
		best_validation_accuracy = 0
		require_improvement = 5000
		last_improvement = 0
		batch_acc_list = []
		val_acc_list = []
		feed_dict_valid = {self.x: X_valid, self.y: y_valid, self.keep_prob: 1.0}
		X_test_1, X_test_2, y_test_1 , y_test_2 = train_test_split(X_test, y_test, test_size=0.5,random_state=5)
		feed_dict_test = {self.x: X_test, self.y: y_test, self.keep_prob: 1.0}
		feed_dict_test_1 = {self.x: X_test_1, self.y: y_test_1, self.keep_prob: 1.0}

		### Train your model here.
		### Calculate and report the accuracy on the training and validation set.
		### Once a final model architecture is selected, 
		### the accuracy on the test set should be calculated and reported as well.
		### Feel free to use as many code cells as needed.

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			num_examples = len(X_train)
		    
			print("training...")
			print()

			for i in range(self.EPOCHS):
				X_train, y_train = shuffle(X_train, y_train)
				for offset in range(0, num_examples, self.BATCH_SIZE):
					end = offset + self.BATCH_SIZE
					batch_x, batch_y = X_train[offset:end], y_train[offset:end]
					sess.run(self.train_operation, feed_dict={self.x:batch_x, self.y:batch_y, self.keep_prob:0.5})

				validation_accuracy = self.evaluate(X_valid, y_valid)
				test_accuracy = self.evaluate(X_test, y_test)
				print("EPOCH {} ... ".format(i+1))
				print("Validation Accuracy = {:.3f}".format(validation_accuracy))
				print("test Accuracy = {:.3f}".format(test_accuracy))
				print()
        
		saver.save(sess,'./lenet')
		print("Model saved")

	def vaild_and_test():
		with tf.Session() as sess:
			saver1.restore(sess=sess, save_path='./lenet')

			acc_test = sess.run(accuracy_operation, feed_dict = feed_dict_test)

			train_accuracy =  evaluate(X_train, y_train)
			validation_accuracy = evaluate(X_valid, y_valid)
			test_accuracy = evaluate(X_test, y_test)
			print("Validation Accuracy = {:.3f}".format(train_accuracy))
			print("test Accuracy = {:.3f}".format(validation_accuracy))
			print()
			msg = " Test acc.: {0:>6.1%} "
			print(msg.format(acc_test))

	def LeNet(self, x):
		#Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
		mu = 0
		sigma = 0.1
		#layer1:Convolutional. Input = 32*32*3. Output = 28*28*32
		conv1_W = tf.Variable(tf.truncated_normal(shape=(5,5,3,32), mean = mu, stddev = sigma))
		conv1_b = tf.Variable(tf.zeros(32))
		conv1 = tf.nn.conv2d(x, conv1_W, strides = [1,1,1,1], padding='VALID') + conv1_b
		#Activation
		conv1 = tf.nn.relu(conv1)
		#Pooling.Input = 28*28*32. Output = 14*14*32
		conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'VALID')
		#dropout
		#     conv1 = tf.nn.dropout(conv1, keep_prob)

		#layer2:Convolutional.Input = 14*14*32. Output = 10*10*64
		conv2_W = tf.Variable(tf.truncated_normal(shape=(5,5,32,64),mean = mu, stddev = sigma))
		conv2_b = tf.Variable(tf.zeros(64))
		conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1,1,1,1], padding = 'VALID') + conv2_b
		#Activation.
		conv2 = tf.nn.relu(conv2)
		#Pooling.Input = 10*10*64. Output = 5*5*64
		conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
		#dropout
		conv2 = tf.nn.dropout(conv2, self.keep_prob)

		#Flatten. for Fully Connected input
		fc0 = flatten(conv2)

		#layer3:Fully Connected.Input = 1600.Output = 800
		fc1_W = tf.Variable(tf.truncated_normal(shape=(1600,800),mean = mu, stddev = sigma))
		fc1_b = tf.Variable(tf.zeros(800))
		fc1 = tf.matmul(fc0, fc1_W) + fc1_b
		#Activation
		fc1 = tf.nn.relu(fc1)
		#dropout
		#     fc1 = tf.nn.dropout(fc1, keep_prob)
		                    
		#layer4:Fully Connected.Input = 800.Output = 400
		fc2_W = tf.Variable(tf.truncated_normal(shape=(800,400),mean = mu, stddev = sigma))
		fc2_b = tf.Variable(tf.zeros(400))
		fc2 = tf.matmul(fc1, fc2_W) + fc2_b
		#Activation
		fc2 = tf.nn.relu(fc2)
		#dropout
		fc2 = tf.nn.dropout(fc2, self.keep_prob)

		#layer5:Fully Connected.Input = 400. Output = 43
		fc3_W = tf.Variable(tf.truncated_normal(shape=(400,43),mean = mu, stddev = sigma))
		fc3_b = tf.Variable(tf.zeros(43))
		logits = tf.matmul(fc2, fc3_W) + fc3_b

		#     logits1 = tf.nn.softmax(logits)
		#     labels_pred_cls = tf.argmax(logits, dimension=1)

		regularizers = (tf.nn.l2_loss(conv1_W)
		            + tf.nn.l2_loss(conv2_W)  + tf.nn.l2_loss(fc1_W) 
		            + tf.nn.l2_loss(fc2_W) + tf.nn.l2_loss(fc3_W))
		                    
		return logits,regularizers
		

