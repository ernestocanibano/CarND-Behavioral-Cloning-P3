# **Behavioral Cloning** 

## Writeup by Ernesto Cañibano

### This is the writeup associated with the third project of the Term 1, of the Self-driving Car Nanodegree

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_model.png "Nvidia Model Modified"
[image2]: ./examples/model_loss.png "Results of the trainning"
[image3]: ./examples/center.png "Center Image"
[image4]: ./examples/flipped.png "Flipped Image"
[image5]: ./examples/lateral.png "Lateral Images"
[image6]: ./examples/recoverings.png "Recoverings Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model is based on the Nvidia paper: [End to End Learning for Self-Driving Cars](CarND-Behavioral-Cloning-P3/examples/end-to-end-dl-using-px.pdf). I began with the same model of the paper but with the size of the input images modified to 160x320 intead 66x200.

![alt text][image1]

The model is detailed in the following table.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| Normalization    		| 160x320x3 RGB image normalized				| 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 78x158x24 	|
| RELU					|												|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 37x77x36   |
| RELU 					|  												|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 19x37x48   |
| RELU 					|  												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 17x35x64   |
| RELU 					|  												|
| Convolution 5x3	    | 1x1 stride, valid padding, outputs 15x33x64   |
| RELU 					|  												|
| Flatten 	    		| outputs 1x31680								| 
| Fully connected		| outputs 1x100        							|
| Fully connected		| outputs 1x50        							|
| Fully connected		| outputs 1x10									|
| Fully connected		| outputs 1x1									|


#### 2. Attempts to reduce overfitting in the model

To avoid overfitting I used the following strategies: 
* Add dropout layers. Finally I discarded this option because altough it improved the overfitting the results with the simulator were worse.
* Collect more data: driving in counter-clockwise and doing recoverings from the left and right sides of the roads. (model.py lines 72 and 73).
* Augment the dataset with: lateral cameras images and flipped images of the center camera. (model.py lines 39 y 61).  

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 114).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, counter-clockwise driving. To improve the perfomance I added images from the lateral cameras and the flipped image of the center camera. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use the convolution neural network model of Nvidia in the same way I already explained in the first point. I thought this model might be appropriate because I read the paper and the purpose of the model was exactly the same of this project, to drive a car using only three cameras.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The dataset only included images of several laps driving the car following the center of the road. I found that my first model had a low mean squared error on the training and validation set. In this case I didn't have problems with overfitting thus I do not need add any dropout layer. 

After that I added images to the dataset to improve the reliability of the model: couter-clockwise driving, flipped images, recovering laps and lateral cameras images.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. I repeat several recovering maneuvers in these points to solve the problem and repeat the process again.

At the end of the process, the vehicle is able to drive autonomously around the track 1 without leaving the road.


#### 2. Final Model Architecture

The final model architecture was the same architecture I already explained in the point one of this writeup but adding a crop layer (model.py line 100). Because of this the size of the interal layers of the model is smaller.

In the next point I will explain the motivation to add the crop layer but comparing the original model and the modified model it is easy to see the big difference of size and process time.

Here is a visualization of the architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| Normalization    		| 160x320x3 RGB image normalized				|
| Crop Layer 	 		| 75x320x3 RGB image normalized				  	|  
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 36x158x24 	|
| RELU					|												|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 16x77x36   |
| RELU 					|  												|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 6x37x48   	|
| RELU 					|  												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 4x35x64   	|
| RELU 					|  												|
| Convolution 5x3	    | 1x1 stride, valid padding, outputs 2x33x64  	|
| RELU 					|  												|
| Flatten 	    		| outputs 1x4224								| 
| Fully connected		| outputs 1x100        							|
| Fully connected		| outputs 1x50        							|
| Fully connected		| outputs 1x10									|
| Fully connected		| outputs 1x1									|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded four laps on track one using center lane driving. Here is an example image of center lane driving and how the image is showed after cropping:

![alt text][image3]

After trainning the model, the validation loss are small but the behaviour of the car in the simulator is not good, the problem of the left turn bias if present. To try to improve it I captured four additional laps driving in counter-clockwise. I added the flipped version of the images of center camera too. 

![alt text][image4]

Trainning the model, and simulating again, the problem was that when car was approaching the edges of the road it finally comes off the road. At this point I decided to use the images of the lateral cams of the previous dataset. Using a big angle correction the car learned to recover the center of the road.
Notice that work with images cropped, where the hood of the car doesn't appear, helps to improve the recovering to the center of the lane.

![alt text][image5]

Until now, with the current dataset, the model works well but in some points of the track the car is unable to recover the center of the lane and gets off of the road. To solve this problem I captured several laps more repeating the recovering maneuvers at these points.
We didn't use the lateral cameras of the captures of the recovering laps because these they got worse the recovering maneuvers.

![alt text][image6]

The resume of the dataset used is the following:
* 3084 captures of normal driving, with flipped images and lateral cameras: 12336 images.
* 2970 captures of counter-clockwise driving, with flipped images and lateral cameras: 11880 images.
* 442 captures of recovery laps, with flipped images: 884 images.
* Total dataset:  25100 images.
* Training data:  20080 images.
* Validation data: 5020 images.

I randomly shuffled the data set and trained the model. In the following graphic it is possible to see the results obtained. 

![alt text][image2]

I needed 18 epochs to train the model. I configured an early stopping callback to stop the train.

With the model chosen and the training performed the car can drive for several laps autonomously in the track 1.
