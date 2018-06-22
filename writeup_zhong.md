# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./wirteup_images/example_images.png "example_images"
[image2]: ./wirteup_images/ImagesDatasetSort.png "ImagesDatasetSort"
[image3]: ./wirteup_images/ImagesDatasetSortBar.png "ImagesDatasetSortBar"
[image4]: ./wirteup_images/ProprocessImages.png "ProprocessImages"
[image5]: ./wirteup_images/AnAugumentedImage.png "AnAugumentedImage"
[image6]: ./wirteup_images/ArgumentImages.png "ArgumentImages"
[image7]: ./wirteup_images/1-Go_straight_or_right.png "Go_straight_or_right"
[image8]: ./wirteup_images/2-No_passing.jpg "No_passing"
[image9]: ./wirteup_images/3-Dangerous_curve_to_the_left.png "Dangerous_curve_to_the_left"
[image10]: ./wirteup_images/4-General_caution.png "General_caution"
[image11]: ./wirteup_images/5-Yield.png "Yield"
[image12]: ./wirteup_images/6-Ahead_only.jpg "Ahead_only"
[image13]: ./wirteup_images/7-Yield.jpg "Yield"
[image14]: ./wirteup_images/8-No_vehicles.png "No_vehicles"
[image15]: ./wirteup_images/9-Keep_left.jpeg "Keep_left"
[image16]: ./wirteup_images/10-Keep_right.png "Keep_right"
[image17]: ./wirteup_images/1-2.png "1-2"
[image18]: ./wirteup_images/3-4.png "3-4"
[image19]: ./wirteup_images/5-6.png "5-6"
[image20]: ./wirteup_images/7-8.png "7-8"
[image21]: ./wirteup_images/9-10.png "9-10"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/MUZUIXIAOHAI/Udacity-P2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is （32, 32, 3）
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

the example images - random choise 25 images

![alt text][image1]

Image data is sorted by number as shown in the figure below

![alt text][image2]

the bar gragh is shown below

![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

for all images include train set、valid set and test set are preprocess used by cv2.equalizeHist

here is an example about the preprocess:

![alt text][image4]

for the train images I transform the images by randomly rotation、shear and translation at last I also add the randomly brightness augment.

Hera is an example of an original image and an augmented iamge:

![alt text][image5]

Here are 25 augmented images - random choise form train set after data agument

![alt text][image6]

I decided to generate additional data because there are more than two thousand photos in category two, there are only more than one hundred in category zero, you could see that at the bar chart 

To add more data to the data set, I randomly assigned one original image to five images , when I train the model , I randomly choise form the data set,so it can deal with data inhomogeneity


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.


My model is a linear model include 2 convolution layer and 3 fully connected layer.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28*28*32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14*14*32					|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10*10*64	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5*5*64					|
| Dropout				| keep_prob:  0.5								|
| Fully connected		| Input = 1600.Output = 800						|
| RELU					|												|
| Fully connected		| Input = 800.Output = 400						|
| RELU					|												|
| Dropout				| keep_prob:  0.5								|
| Output		 		| Input = 400.Output = 43						|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I shuffle the train data set for each epochs, the batch size is 128 , number of epochs is 15 ,learning rate is 0.001.

My model has achieved good accuracy when it is less than 10 epochs, and the batch size is small enought for the model.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.8%
* validation set accuracy of 97.7%
* test set accuracy of 96.2%



* What architecture was chosen?

I choose the LeNet architecture.

* Why did you believe it would be relevant to the traffic sign application?

Because it is a classical architecture for images.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

There are a high accuracy on the training , validation and test set ,99.8%,97.7%,96.2%. 

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11]
![alt text][image12] ![alt text][image13] ![alt text][image14] ![alt text][image15] ![alt text][image16]

The 6nd image might be difficult to classify because the road sign only take up a small part of the picture, and the picture is not cut.

the 8th image might be difficult to classify because the road sign is similar to the No passing road sign ,and the image's background is black.

other images work well for the model.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			      		  |     Prediction	        					| 
|:---------------------------:|:-------------------------------------------:| 
| Go straight or right		  | Go straight or right						| 
| No passing    			  | No passing 									|
| Dangerous curve to the left | Dangerous curve to the left 				|
| General caution       	  | General caution 				 			|
| Yield						  | Yield    									|
| Ahead only 		  		  | Speed limit (30km/h)						| 
| Yield  			  		  | Yield										|
| No vehicles 				  | No passing  								|
| Keep left        	  		  | Keep left  				 					|
| Keep right 				  | Keep right   								|


The model was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 96.2%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

This part's code is located in penultimate cell of the Ipython notebook.

For the 1st and 2nd image, The top five soft max probabilities you cloud see at the below figure.

![alt text][image17]

For the 3rd and 4th image, The top five soft max probabilities you cloud see at the below figure.

![alt text][image18]

For the 5th and 6th image, The top five soft max probabilities you cloud see at the below figure.

![alt text][image19]

For the 7th and 8th image, The top five soft max probabilities you cloud see at the below figure.

![alt text][image20]

For the 9th and 10th image, The top five soft max probabilities you cloud see at the below figure.

![alt text][image21]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


