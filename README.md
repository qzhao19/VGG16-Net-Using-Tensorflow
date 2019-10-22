# VGG16-Net-Using-Tensorflow
Implementation of vgg16 network

# Overview

In this repertoire, I have implemented Vgg16 network using tensorflow. Vgg16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition". It performs on the dataset: 16 Flowers, which presents 17 category flower dataset with 80 images for each class. The images have large scale, pose and light variations and there are also classes with large varations of images within the class and close similarity to other classes. 

Vgg16 network achieved 92.7% top-5 test accuracy in ImageNet, which stores over 14 million images belonging to 1000 classes. The network architecture presents image below:

![image](https://github.com/zhaoqi19/VGG16-Net-Using-Tensorflow/blob/master/image/vgg16.png)

<p align="center">
	<img src="https://github.com/zhaoqi19/VGG16-Net-Using-Tensorflow/blob/master/image/vgg16.png"  width="250" height="140">
	<p align="center">
		<em>图片示例2</em>
	</p>
</p>


For more details on the underlying model please refer to the following paper:

    @article{simonyan2014very,
      title={Very deep convolutional networks for large-scale image recognition},
      author={Simonyan, Karen and Zisserman, Andrew},
      journal={arXiv preprint arXiv:1409.1556},
      year={2014}}
      
 # Requirement

- Python 3.6
- NumPy 1.16.5
- Tensorflow >= 1.14
- OpenCV >= 3.4.3
- matplotlib >= 3.1.1

# Contents

- `model.py`: Class with the graph definition of the Vgg16.
- `layers.py`: Neuron network layers containing convolutional layer, full collection layer, batch normalization and maxpooling.
- `evals.py`: Model evaluation's function containing calc_loss_acc and train_op.
- `train.py`: Script to run the training process.
- `images/*`: contains three example images.
- `outputs`: output result folder containing two sub-folder (accuracy_loss and model)
