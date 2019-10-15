import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def load_label_file(file_path, shuffle=True):
    """load label file from a txt file, returns a image filename list and label list
    """
    images_list = []
    labels_list = []
    
    # open image and label file
    with open(file_path, 'r') as file:
        lines = [line.strip().split(' ') for line in file.readlines()]
    # shuffle paths
    if shuffle:
        random.shuffle(lines)
    
    for i in range(len(lines)):
        images_list.append(lines[i][0])
        labels_list.append(lines[i][1])
    # get base file path
    base_path = os.path.join(os.getcwd(), 'test_dataset', 'images')
    #complet image file path with cwd base path
    image_paths_list = [os.path.join(base_path, image_list) for image_list in images_list]
    return image_paths_list, labels_list


def read_image(image_path, image_shape=(64, 64, 3)):
    """The function is to read image
    Args:
        image_path: a string, containing image file path
    Return:
        image object with default uint8 type [0, 255]
    
    """
    bgr_image = cv2.imread(image_path)
    # resize image
    bgr_image = cv2.resize(bgr_image, (image_shape[0], image_shape[1]), interpolation=cv2.INTER_NEAREST)
    
    # make sure image with 3 channel
    if len(bgr_image.shape) != 3:
        print('Warning: grey image!', image_path)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    # convert bgr image into a rgb image
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    return np.asarray(rgb_image)


def preprocessing_image(images, mode='tf'):
    """image preprocessing, Zero-center by mean pixel, it will scale pixels between -1 and 1.
    Args:
        images: 4D image or tensor with [batch_size, height, width, channel]
        mode: One of "tf", "torch" or "caffe".
            - caffe:then will zero-center each color channel with respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.
            - torch: will scale pixels between 0 and 1 and then will normalize each channel with respect to the ImageNet dataset.

    """
    if images.get_shape().ndims != 3:
        raise ValueError('Input must have size [height, width, channel>0]')
    
    if mode == 'tf':
        images /= 127.5
        images -= 1.0

    elif mode == 'torch':
        images /= 255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_channels = images.get_shape().as_list()[-1]
        channels = tf.split(images, num_or_size_splits=num_channels, axis=2)
        for i in range(num_channels):
            channels[i] -= mean[i]
            channels[i] /= std[i]
        images = tf.concat(channels, axis=2)

    elif mode == 'caffe':
        mean = [123.68, 116.779, 103.939]
        num_channels = images.get_shape().as_list()[-1]
        channels = tf.split(images, num_or_size_splits=num_channels, axis=2)
        for i in range(num_channels):
            channels[i] -= mean[i]
        images = tf.concat(channels, axis=2)
    else:
        raise ValueError('Preprocessing image mode should be one of 3 method (tf, torch, caffe)')

    return images


def show_image(image, title=None):
    """show image 
    """
    plt.imshow(image)
    plt.axis('on')    
    plt.title(title) 
    plt.show()