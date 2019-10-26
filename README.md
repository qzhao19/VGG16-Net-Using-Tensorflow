# VGG16-Net-Using-Tensorflow
Implementation of vgg16 network

# Overview

In this repertoire, I have implemented Vgg16 network using tensorflow. Vgg16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition". It performs on the dataset: 16 Flowers, which presents 17 category flower dataset with 80 images for each class. The images have large scale, pose and light variations and there are also classes with large varations of images within the class and close similarity to other classes. 

Vgg16 network achieved 92.7% top-5 test accuracy in ImageNet, which stores over 14 million images belonging to 1000 classes. The network architecture presents image below:


<p align="center">
	<img src="https://github.com/zhaoqi19/VGG16-Net-Using-Tensorflow/blob/master/image/vgg16.png"  width="560" height="400">
	<p align="center">
		<em>Vgg16 architecture</em>
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
- `images/*`: contains example images.
- `17_flowers/*`
	- `images/*` : dataset directory (you should put the 17flowers dataset into here)
	- `tensorboard_dir/*` : tensorboard logs path
		- `train`
		- `valid` 
	- `tfrecords/*` : tfrecords file directory
		- `train_tfrecords/*`
		- `valid_tfrecords/*`
	- `train_valid_file/*` 
	- `vgg16_model/*` 
- `utils`:
	- `build_label_file.py` : create the label_file to read image and label
	- `processing_image.py` : processe image 
	- `tfrecords.py` : create tfrecords file and read tfrecords file
	- `tfrecords.py` : test tfrecords.py file 

# Usages

First, I strongly recommend to take a look at the entire code of this repository. In fact, we need to download the 17 flowers dataset directly from [here](https://github.com/ck196/tensorflow-alexnet/blob/master/17flowers.tar.gz). A local training job can be run with the following command:

    python train.py \
		--valid_steps = 11 \
		--max_steps = 1001 \
		--batch_size = 128 \
		--base_learning_rate = 0.001 \
		--input_shape = 784 \
		--num_classes = 10 \
		--keep_prob = 0.75 \
		--image_height = 64 \
		--image_width = 64 \
		--image_channel = 3 \
		--mode = 'tf' \
		--train_tfrecords_file = './17_flowers/tfrecords/train_tfrecords.tfrecords' \
		--valid_tfrecords_file = './17_flowers/tfrecords/valid_tfrecords.tfrecords' \
		--tensorboard_dir = './17_flowers/tensorboard_logs/' \
		--saver_dir = './17_flowers/vgg16_model/'
