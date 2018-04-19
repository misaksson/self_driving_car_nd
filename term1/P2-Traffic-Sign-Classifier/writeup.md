**Traffic Sign Recognition**

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

[image1]: ./writeup_images/exploratory_visualization.png "Exploratory Visualization"
[image2]: ./writeup_images/normalization.png "Normalization"
[image3]: ./data/additional/label4.jpg "Traffic Sign 1"
[image4]: ./data/additional/label7.jpg "Traffic Sign 2"
[image5]: ./data/additional/label12.jpg "Traffic Sign 3"
[image6]: ./data/additional/label17.jpg "Traffic Sign 4"
[image7]: ./data/additional/label25.jpg "Traffic Sign 5"

## Rubric Points
###Here I consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

#### Files Submitted
##### Submission Files
All files are submitted except:

* The training-, validation- and test-set, which however can be downloaded by running the provided setup.py.
* The tensorflow saved-model, which is too big for github. Similar result can however be achieved by running the code in the Jupyter notebook, then copy the model from training output to ./model/*

#### Dataset Exploration
##### Dataset Summary
The TrafficSignData module extracts the requested values.

##### Exploratory Visualization
The TrafficSignData module show examples of each traffic sign label, along with the number of examples of each label in the training set.

#### Design and Test a Model Architecture
##### Preprocessing
I did only normalize the image data in the same way as was proposed in the description. There was also suggestions to gray-scale convert the image but I didn't see the point in doing that since it only removes information, and I think that's bad when there are no limits specified for the size of the model (e.g. in terms of number of weights or FLOPS).

##### Model Architecture
This is described both in Jupter notebook and this document.

##### Model Training
This is described both in Jupter notebook and this document.

##### Solution Approach
This is described both in Jupter notebook and this document.

#### Test a Model on New Images
##### Acquiring New Images
There are five new images provided and some discussion about what's making it hard to classify them.

##### Performance on New Images
The performance is provided and compared to the test set.

##### Model Certainty - Softmax Probabilities
The top 5 soft-max score is measured for all new images and presented both in the Jupyter notebook and this document.

---
### Writeup / README

#### 1. Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it! and here is a link to my [project code](https://github.com/misaksson/self_driving_car_nd/tree/master/term1/CarND-Traffic-Sign-Classifier-Project/)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It shows one example of each of the traffic signs in the data-set, along with the number of examples available in the training-set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Image data preprocessing.

As a first step, I decided to normalize the images to the range [-1, 1]. This was done in the same way for all images, just mapping the input range [0, 255] to [-1, 1]. The neural network needs it input centered, and close to 0 for effective training. I think it might have been better to instead normalize the images using the mean and standard deviation of each image, since that probably would make the model more robust for shifting light conditions.

Here is an example of a traffic sign image and its histogram before and after normalization.

![alt text][image2]

I also decided to generate additional training data because the number of training examples was imbalanced, ranging from 180 to 2010 examples per class.

To add more data to the the data set, I oversampled the class-labels having fewer examples. I did not implement any augmentation of the oversampled images, they are just duplicated until the number of samples are equal. The reason that I didn't do any augmentation on these images was to avoid producing "water-marks" in the images e.g. at the image borders, which might make such errors necessary for the classifier to believe the image belongs to an less represented class-labels.

Instead the plan was to do augmentation in the next step and on all images of the oversampled set, which would balance the "water-mark" error equally for all classes. Unfortunately I didn't have time to test this.


#### 2. Final model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 7x7     	| 1x1 stride, valid padding, outputs 26x26x15 	|
| RELU6					|												|
| Max pooling 2x2       | 2x2 stride,  outputs 13x13x15                 |
| Convolution 6x6       | 1x1 stride, valid padding, outputs 8x8x49     |
| ELU                   |                                               |
| Max pooling 2x2     	| 2x2 stride,  outputs 4x4x49       			|
| Fully connected		| outputs 8362        							|
| CRELU                 |                                               |
| Fully connected       | outputs 811                                   |
| Dropout               | keep_prob 0.5                                 |
| Sigmoid               |                                               |
| Fully connected       | outputs 43                                    |


#### 3. Training my model.

I did some experiments with the learning rate, trying to find a small value, that still was large enough to not get stuck at local minimums. The value I ended up with was 0.0004. If I would spend more time on this then I think implementing the learning rate using a tensor variable might be a good idea, since it makes it possible to use learning rate momentum.

I did not care much about the number of epochs, instead I just saved the model every time the validation accuracy reached a new maximum. By doing this, I somewhat avoid the risk of continuing the training too long, which would make the model overfit the training data and perform poorly at any unknown data.

#### 4. The approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.962
* test set accuracy of 0.944

This is calculated in the Jupyter notebook cell 15, 16 and 17.

The model was trained using a neural-network training framework that I implemented my self for this project. This allowed me to easily generate variants of the graph layout and its parameter settings, automatically alternating the configuration for each training session such that all possible permutations of the parameters are trained and evaluated. Unfortunately it took some time to implement this, so there was no time in the end to use it for any exhaustively evaluation of lots of settings. Instead I based my model on the LeNet-5 and tried some variants.

##### Activation functions
I experimented with different activation functions. The result of this was that I ended up using different activation functions for each of the four layers (RELU6, ELU, CRELU and Sigmoid).

##### Convolutional filter size
I also evaluated different filter sizes in the convolutional layers, ending up with best result using 7x7 resp. 6x6. I think removing the max-pool layers and trying larger filter sizes might give better result. Probably also adding more output channels.

##### L2 loss regularizatoin
Another thing I was experimenting with was L2 loss regularization, which helps to avoid overfitting the model. In the final model this did however produce worse result, so I set the beta value to 0.

##### Dropout
Dropout makes the model more redundant since it can't rely on any particular data path. This did show good result and is included in the final model with a keep probability of 0.5, which however is the only value I tested.

##### Penalize frequently represented class-labels
By looking at the confusion matrix from an evaluation of the validation set, it was obvious that the unequal number of training examples for the different classes caused problems. In an attempt to counter this, I tried to penalize frequent class-labels by multiplying the output from the logits with a weighed class vector, that is before the softmax_cross_entropy operation. I was however unable to find a weight vector that produced good result with this method.

##### Oversampling training data
Another method I tried to compensate for the imbalanced training data was to apply oversampling, where I simply duplicated the images of less frequent class-labels until the numbers are equal. This was successful and made the accuracy increase from around 0.94 to 0.96+.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5]
![alt text][image6] ![alt text][image7]

Non of the images I selected was very hard, they are all more or less directly from the front and the light conditions seems fine. The background are not too annoying, and I didn't choose any of the less frequently represented images, or the ones that almost look the same.

But in general, speed signs might be hard because there are several types of them that only differ by the number. There are also several warning signs that look somewhat similar. The priority road sign is however very unique and should typically not be of any problem.

#### 2. Predictions on the new traffic signs compared to the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 70 km/h        		| 70 km/h   									|
| 100 km/h     			| 100 km/h 										|
| Priority road			| Priority road									|
| No entry	      		| No entry     					 				|
| Road work			    | Road work          							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This is better than the test set where the accuracy was 94.4%.

#### 3. How certain is the model when predicting on each of the five new images.

The code for making predictions on my final model is located in the 22th cell of the Jupyter notebook.

The model is very certain for all images. All predictions are correct and the soft-max probability is never less than 0.95 (for the 70 km/h sign, all the other signs have probabilities above 0.98).

##### Image 1 (70 km/h)

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .95         			| 70 km/h   									|
| .03     				| 20 km/h 										|
| .01					| 30 km/h										|
| .00	      			| General caution				 				|
| .00				    | 60 km/h      	           						|


##### Image 2 (100 km/h)

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| .98                   | 100 km/h                                      |
| .007                  | Vehicles over 3.5 metric tons prohibited      |
| .005                  | 80 km/h                                       |
| .000                  | 120 km/                                       |
| .000                  | No passing for vehicles over 3.5 metric tons  |


##### Image 3 (Priority road)

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.00                  | Priority road                                 |
| .000                  | No passing for vehicles over 3.5 metric tons  |
| .000                  | Traffic signals                               |
| .000                  | Stop                                          |
| .000                  | No entry                                      |


##### Image 4 (No entry)

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| .997                  | No entry                                      |
| .002                  | Stop                                          |
| .000                  | No passing                                    |
| .000                  | Yield                                         |
| .000                  | 60 km/h                                       |


##### Image 5 (Road work)

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| 1.00                  | Road work                                     |
| .000                  | Beware of ice/snow                            |
| .000                  | Dangerous curve to the right                  |
| .000                  | Right-of-way at the next intersection         |
| .000                  | Pedestrians                                   |
