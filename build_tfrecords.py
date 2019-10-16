from utils.tfrecords import write_tfrecords, read_tfrecords, get_example_nums

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_tfrecords_file', './17_flowers/tfrecords/train_tfrecords.tfrecords', 'Training tfrecords data path')

flags.DEFINE_string('valid_tfrecords_file', './17_flowers/tfrecords/valid_tfrecords.tfrecords', 'Validation tfrecords data path')

flags.DEFINE_string('train_file', './17_flowers/train_valid_file/train_file.txt', 'The path to training file names')

flags.DEFINE_string('valid_file', './17_flowers/train_valid_file/valid_file.txt', 'The path to validation file names')

flags.DEFINE_integer('image_height', 64, 'The image height')

flags.DEFINE_integer('image_width', 64, 'The image width')

flags.DEFINE_integer('image_channel', 3, 'The image channels')

FLAGS = flags.FLAGS



if __name__ == '__main__':
    # set parameters
    train_file = FLAGS.train_file
    valid_file = FLAGS.valid_file
    
    train_tfrecords_file = FLAGS.train_tfrecords_file
    valid_tfrecords_file = FLAGS.valid_tfrecords_file
    
    image_shape = (FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel)
    
    # generate train_data.tfrecords
   
    write_tfrecords(train_file, image_shape, train_tfrecords_file)
    nums_train_example = get_example_nums(train_tfrecords_file)
    print("Train data tf_example nums = {}".format(nums_train_example))

    # generate valid_data.tfrecords
    
    write_tfrecords(valid_file, image_shape, valid_tfrecords_file)
    nums_valid_example = get_example_nums(valid_tfrecords_file)
    print("Train data tf_example nums = {}".format(nums_valid_example))
    
    
