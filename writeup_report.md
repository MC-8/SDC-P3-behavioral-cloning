# **Behavioral Cloning** 

## Project writeup


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[centerdrive]: ./report_images/center.jpg "Center Image"
[leftdrive]: ./report_images/left.jpg "Left Image"
[rightdrive]: ./report_images/right.jpg "Right Image"
[recovery]: ./report_images/recovery.jpg "Recovery Image"
[centercropped]: ./report_images/center_cropped.jpg "Cropped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network adapted from (https://devblogs.nvidia.com/deep-learning-self-driving-cars/).    

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 17). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 80).
The number of epochs eventually selected is low (2), this is because an higher number of epochs would increase the validation loss, despite a decrease in the training loss, which is a sign of overfitting.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. The recovery driving was particularly needed for sharp turns, where the car would often go straight or to a wrong direction. By recording data starting with the car in a wrong position, then correcting the trajectory, the car would eventually learn how to recover from bad trajectories. However, some recovery behaviour, such as sharp turns or oscillatory steering, can be observed during normal driving on some turns, which may be due to excessively conservative recovery procedure.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use existing networks and train them to adapt to this self-driving problem.

My first step was to use a convolution neural network model similar to the LeNet architecture, since I already had an implementation for it and if it is good at detecting traffic signs and numbers, it may be good at detecting road edges as well.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that with am high number of training epochs the model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I reduced the number of epochs. I also tried to introduce two dropout layers (50% probability) but the improvement was marginal so I decided to try the network developed by NVIDIA (linked above) because it is a proven network on autonomous vehicles.

With just two epochs of training I obtained very low training loss and validation loss which gave me confidence that the network might work very well.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, diving straight to the lake, or going for a pic-nic in the forest. To improve the driving behaviour I recorded multiple "snips" of recovering from a bad trajectory, and repeating challenging turns multiple times, as well as driving the circuit on the opposite direction, to make sure the network would be as general as possible.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. As it can be seen from the video.mp4.

#### 2. Final Model Architecture

The final model architecture (model.py lines 64-81) consisted of a convolution neural network with the following layers and layer sizes.

| Layer         		|     Description	        					    | 
|:---------------------:|:-------------------------------------------------:| 
| Input             | 160x320x3 RGB image   						        | 
| Cropping          | Trim top 65 and bottom 25 pixels, outputs 90x320x3    |
| Lambda            | Normalizes image data to have values in [-0.5:0.5]    |
| Convolution 5x5   | 2x2 stride, valid padding, outputs 33x158x24 	        |
| RELU				|												        |
| Convolution 5x5   | 2x2 stride, valid padding, outputs 15x77x36 	        |
| RELU			    |												        |
| Convolution 5x5   | 2x2 stride, valid padding, outputs 6x37x48 	        |
| RELU			    |												        |
| Convolution 3x3   | 1x1 stride, valid padding, outputs 4x35x64 	        |
| RELU			    |												        |
| Convolution 3x3   | 1x1 stride, valid padding, outputs 2x33x64 	        |
| RELU			    |												        |
| Fully connected	| outputs 100        							        |
| Fully connected	| outputs 50        							        |
| Fully connected	| outputs 10        							        |
| Fully connected	| outputs 1         							        |
| Output    		| Vehicle steering                                      | 

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][centerdrive]

Additionally I've used images coming from the left and right cameras.

Left:
![alt text][leftdrive]

Right:
![alt text][rightdrive]

This was to augment the dataset and to train the network to handle the situation where the car may be too close to the edges of the road.
In order to properly train the network, it was necessary to adjust the steering angle of the left and right image, so that the "correct" steering angle would be the steering angle from the central image, but corrected by a positive (left image) or negative (right image) value.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to the center of the road if it sways away. This image show what a recovery looks like starting from a position almost out of the track.

![alt text][recovery]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also drove around the circuit on the other way around. The alternative would be to flip the images in the dataset (and invert the steering angle sign) but I opted for a more realistic approach.
I also tried to improve the dataset by driving in the second circuit but the overall driving performances were not so great.
As it is now, the car can drive very well in the first circuit, but not so well on the second one, so the model is not very general, and the car will only be able to drive reliably on tracks with the same characteristics to the first one (mostly flat, with mostly yellow road edges).

After the collection process, I had about 29000  data points. I cropped the images to remove irrelevant parts (for lane keeping) and the images fed to the network look like the following:
![alt text][centercropped]


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as explained earlier. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 4. Conclusion
It was a fun project to execute (especially the driving part) that taught me how a dataset can be captured in simulation and some of the techniques to obtain a fairly robust driving. Driving in a more complex enviroment such as the second track proved to be rather challenging: sharp turns, steep slopes with very low visibility, unclear lane boundaries, are all challenges that would require crafting a better training set. I could have performed a more thorough investigation on network architecture and image (pre)processing, but I am satisfied for now, with having learned the basics of behavioral cloning.