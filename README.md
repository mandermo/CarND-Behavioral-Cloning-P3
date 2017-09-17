#**Behavioral Cloning** 

##About

###My solution to the "Behaviorial Cloning Project" project in the [Udacity Self-Driving Car Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013). Here you can find a [link to Udacity's upstream project](https://github.com/udacity/CarND-Behavioral-Cloning-P3).

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* README.md this file summarizing the results
* writeup_report no file with this name, README.md is used instead
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (mo
* model.h5 containing a trained convolution neural network 
* common.py common code for drive.py and model.py, used preprocessing of the data
* visualizenetwork.py used to visualize convolutional layers
* manualdrivefilter.py used to filter out frames to keep from video, for recovery driving
* video.mp4 a video recording of the car driving around a full lap around the track

To visualize the convolutional layers, do this if you use docker:
for i in {1..5}; do rm -rf activ${i}; mkdir activ${i}; docker run -it --rm -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit python visualizenetwork.py model.h5 conv${i} udacity-my-driving-data/IMG/center_2017_08_23_09_26_03_711.jpg --todir activ${i}; done

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
./rundrive.sh
```
which executes drive.py using docker with model.h5 as argument.

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a five convolutional layers with 5x5 and 3x3 filter sizes. I use the elu activation function for the convolutions and for the fully connected layers to introduce nonlinearity. The data in normalized in a preprocessing stage before it is feed to the network. It is done both for model.py, drive.py and visualizenetwork.py. The normalization code is in common.py.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on data from four laps to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer. The adam optimizer had some artifacts for my model. Sometimes after getting very low loss (the validation loss was also small), then the loss shoots up to a very high value eg.
```
10288/10288 [==============================] - 190s - loss: 0.0376 - val_loss: 0.0412
Epoch 7/7
10288/10288 [==============================] - 176s - loss: 55153.9662 - val_loss: 1.4536

```
I saw this happening on the third epoch also. Here I found an explanation for it: https://stackoverflow.com/a/42420014. I tweaked the epsilon parameter as suggested in the stackoverflow post, but I could also have switched from Adam.

####4. Appropriate training data

I did three drives where I tried to stay in center, one of them reverse. For teaching it to recover when it goes out of center I didn't use the right and left camera with artificial constructed steering angles, as suggested in course, just because I feelt it is to heuristicish. I did a drive where I wiggled around the road and used a program I wrote manualdrivefilter.py that I use to construct a new driving log with only the frames I want. I choose to keep the frame where I turning back to center and skip the frames when I drive away from center. The program works by showing the frames and the steering angle and then you decide which frames to keep. You toggle keeping and skipping frames with the space bar.

###Model Architecture and Training Strategy

####1. Solution Design Approach

I started with a network without any hidden layer, just flattening and the output angle node. It sways around the road and then got stuck. I augmented the data by flipping it.

When I added support for training on multiple drives, then I changed the code so it randomly flips the frames, then I forgot to also flip the steering angle. For some reason it still managed to drive part of the track. I tried out layers with three convolutions, some max pooling and a locally connected layer. It didn't manage around the track. I later went back in the git history and fixed the angle bug and it drove in to water before brige anyway.

I switched to the Nvidia architecture with YUV instead of RGB input. I choose YUV because the Nvidia description said they did it https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/. I choose to use Elu instead of ReLU because I got information in in feed back in last assignment that Elu generally train better than ReLU.

I still had not found my angle bug at this point. I though it was overfitting (validation loss didn't fall as fast as loss), so I added dropout after the first convolution as and the fifth with a probablility of 30%. It managed to drive past bridge with luck. Found the angle bug and fixed it, after that it drives around the track. Backported the fix to before I added dropout and it drove around 3/4 of the track before driving off the road.

The Nvidia description writes that they don't predict the steering angle, but the inverse steering radius. I tried to do the same for fun, but it steers very agressively and ends up driving in a small circle. I didn't figure out exactly why, but commited the code for that in a separate branch called steeringradius.

####2. Final Model Architecture

It is the Nvidia architecture. With added dropout after the first and the fifth convolution, with probability 30%. I guess dropout is not needed for Nvidia, because they sit on huge training data. It is five convolutions and three fully connected layers.

####3. Creation of the Training Set & Training Process

See "Appropriate training data" section for info about traning process.