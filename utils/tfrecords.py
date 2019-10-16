import os
import tensorflow as tf
from build_dataset import load_label_file
from processing_image import read_image, preprocessing_image


def int64_feature(value):
    """create int64 type feature 
    """
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    """create bytes type feature 
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tfrecords(file_path, image_shape, tfrecord_file):
    """Function takes label and image height, width, channel etc, to store as tfrecord data 
    Args:
        file_path: the path to image_label_file
        image_size: int tuple, containing image height and image width
        output_tfrecords_dir: the path points output_tfrecords_dir
    Raises:
        ValueError: 
        
    """
    # get images and labels list, and make sure the number of images and labels is equal
    image_paths_list, labels_list = load_label_file(file_path)
    if len(image_paths_list) != len(labels_list):
        raise ValueError('The size of image dose not match with that of label.')
    
    # create a tfrecord writer
    tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_file + '.tfrecords')
    for i, [image_path, label] in enumerate(zip(image_paths_list, labels_list)):
        # determine image path whether exists 
        if not os.path.exists(image_path):
            raise ValueError('Error: %s not exists' %image_path)
            continue
        # check image format
        if image_path.split('/')[-1].split('.')[-1] not in ['jpg', 'JPEG']:
            raise ValueError('The format of image must be jpg or JEPG')
            continue
        # read image and convert it into bytes data
        image = read_image(image_path, image_shape)
        image_bytes = image.tostring()
        if i%100 == 0 or i == (len(image_paths_list)-1):
            print('------------processing:%d-th-image-----------' %(i))
            print('current image_path=%s' %(image_path),'shape:{}'.format(image.shape),'labels:{}'.format(label))
        
        feature_dict = {}
        tf_example = tf.train.Example(features=tf.train.Features(feature={'image_height': int64_feature(image_shape[0]), 
                                                                          'image_width': int64_feature(image_shape[1]), 
                                                                          'image_depth': int64_feature(image_shape[2]),
                                                                          'image': bytes_feature(image_bytes),
                                                                          'label': int64_feature(int(label))}))
        try:
            tfrecord_writer.write(tf_example.SerializeToString())
            # print('TF record data has been done')
        except ValueError:
            print('Invalid example, ignoring')
            
    tfrecord_writer.close()



def read_tfrecords(tfrecord_file, image_shape=(64, 64, 3)):
    """Read tf records file
    Args:
        tfrecords_path: string, output tensorflow records data path
        preprocessing: bool, maje sure that we need to preprocessing image
    Returns:
        tf image with float32 data and tf label with int32
    """
    # create filename queue, it does not limit input number
    filename_queue = tf.train.string_input_producer([tfrecord_file])
    
    # initialize tfrecord reader
    tfrecords_reader = tf.TFRecordReader()
    # tfrecords_reader load a serialized tf example
    _, serialized_example = tfrecords_reader.read(filename_queue)
    # get features from serialized tf example
    features = tf.parse_single_example(serialized_example, features={'image_height': tf.FixedLenFeature([], tf.int64), 
                                                                     'image_width': tf.FixedLenFeature([], tf.int64), 
                                                                     'image_depth': tf.FixedLenFeature([], tf.int64),
                                                                     'image': tf.FixedLenFeature([], tf.string),
                                                                     'label': tf.FixedLenFeature([], tf.int64)})
    # get all info about image and label
    image_height = features['image_height']
    image_width = features['image_width']
    image_depth = features['image_depth']
    # get image raw data
    images = tf.decode_raw(features['image'], out_type=tf.uint8)
    # reshape image
    images = tf.reshape(images, shape=[image_shape[0], image_shape[1], image_shape[2]])
    images = tf.cast(images, dtype=tf.float32)
    
    labels = tf.cast(features['label'], tf.int32)
    
    return images, labels



def generate_image_label_batch(images, labels, batch_size, num_classes, one_hot=True, shuffle=True):
    """Construct a queued batch of images and labels.
    Args:
        image: 3D Tensor of [height, width, 3] of type.float32.
        label: int32
        in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        one_hot: bool, one hot encode
        shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels:
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    # min_queue_examples: int32, minimum number of samples to retain
    num_threads = 1
    min_after_dequeue = 200
    capacity = min_after_dequeue + 3 * batch_size
    if shuffle:
        images_batch, labels_batch = tf.train.shuffle_batch([images, labels], 
                                                            batch_size=batch_size,
                                                            capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue,
                                                            num_threads=num_threads)
    else:
        images_batch, labels_batch = tf.train.batch([images, labels],
                                                    batch_size=batch_size,
                                                    capacity=capacity,
                                                    num_threads=num_threads)
    if one_hot:
        labels_batch = tf.one_hot(labels_batch, depth=num_classes, on_value=1.0, off_value=0.0)

    # Display the training images in the visualizer.
    return images_batch, labels_batch

