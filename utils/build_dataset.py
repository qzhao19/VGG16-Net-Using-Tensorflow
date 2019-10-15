import os
import random
import numpy as np


def create_label_file(dir_path):
    """create a txt file containing image filepath and its label
    Args:
        dir_path: directory path to image root dir
    """
    image_label_file = []
    dir_names = os.listdir(dir_path)
    for dir_name in dir_names:
        image_names = os.listdir(os.path.join(dir_path, dir_name))
        for image_name in image_names:
            image_path = os.path.join('./17_flowers_dataset', 'images', dir_name, image_name)
            image_label_file.append([image_path, dir_name])
    image_label_file = np.array(image_label_file)
    
    try:
        np.savetxt('file.txt', image_label_file, delimiter=' ', fmt='%s')
        print('Label file has bee done')
    except Exception as error:
        print(error)



def load_label_file(file_path, shuffle=True):
    """load label file from a txt file, returns a image filename list and label list
    Args:
        file_path: label file path that is created from creat_label_file function 
    """
    image_paths_list = []
    labels_list = []
    
    # open image and label file
    with open(file_path, 'r') as file:
        lines = [line.strip().split(' ') for line in file.readlines()]
    # shuffle paths
    if shuffle:
        random.shuffle(lines)
    
    for i in range(len(lines)):
        image_paths_list.append(lines[i][0])
        labels_list.append(lines[i][1])
    
    return image_paths_list, labels_list


def create_train_valid_set(file_path):
    """create train and validation dataset 
    """
    with open(file_path, 'r') as file:
        lines = [line.strip().split(' ') for line in file.readlines()]
    
    train_data = lines[0:int(len(lines)*0.75)]
    valid_data = lines[int(len(lines)*0.75):len(lines)]

    return train_data, valid_data
