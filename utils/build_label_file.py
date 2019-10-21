import os
import random
import numpy as np


def write_label_file(dir_path, output_path, shuffle=True):
    """create train and validation data file 
    """
    if not os.path.exists(dir_path):
        raise ValueError('Please check dataset file path.')

    if not os.path.exists(output_path):
        os.makedirs('./17_flowers/train_valid_file')


    image_label_file = []
    dir_names = os.listdir(dir_path)
    for dir_name in dir_names:
        image_names = os.listdir(os.path.join(dir_path, dir_name))
        for image_name in image_names:
            image_path = os.path.join(dir_path, dir_name, image_name)
            image_label_file.append([image_path, dir_name])
    
    # shuffle paths
    if shuffle:
        random.shuffle(image_label_file)
    
    train_data = image_label_file[0:int(len(image_label_file)*0.75)]
    valid_data = image_label_file[int(len(image_label_file)*0.75):len(image_label_file)]

    train_file = np.array(train_data)
    valid_file = np.array(valid_data)

    try:
        np.savetxt(os.path.join(output_path, 'train_file.txt'), train_file, delimiter=' ', fmt='%s') 
        np.savetxt(os.path.join(output_path, 'valid_file.txt'), valid_file, delimiter=' ', fmt='%s')   
        print('Train data and validation data file has bee done')
    except Exception as error:
        print(error)

# create_label_file('./17_flowers/images', './17_flowers/train_valid_file')


def load_label_file(file_path):
    """load label file from a txt file, returns a image filepath list and label list
    Args:
        file_path: train data or validation file path for default mode 'train' and 'validation'
    Returns:
        image paths list and image label list
    """
    image_paths_list = []
    labels_list = []
    
    if not os.path.exists(file_path):
        raise ValueError('Please check dataset file path.')

    # open image and label file
    with open(file_path, 'r') as file:
        lines = [line.strip().split(' ') for line in file.readlines()]

    for i in range(len(lines)):
        image_paths_list.append(lines[i][0])
        labels_list.append(lines[i][1])
    
    return image_paths_list, labels_list

