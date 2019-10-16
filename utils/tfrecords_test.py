import tensorflow as tf
from processing_image import preprocessing_image, show_image
from tfrecords import read_tfrecords, generate_image_label_batch

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('tfrecord_file', './17_flowers/tfrecords/valid_tfrecords.tfrecords', 'The path to saved checkpoints')

flags.DEFINE_integer('batch_size', 10, 'The number of images in each batch during training')

flags.DEFINE_integer('num_classes', 13, 'The number of label classes')


FLAGS = flags.FLAGS

def test_tfrecords(tfrecords_file):
    """
    """
    tf_images, tf_labels = read_tfrecords(tfrecords_file)
    # tf_images = preprocessing_image(tf_images, mode)
    # images_batch, labels_batch = generate_image_label_batch(tf_images, tf_labels, batch_size=batch_size, num_classes=num_classes)
    # open a session 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(4):
            images, labels = sess.run([tf_images, tf_labels])
            print('image shape: {}, image tpye: {}, label type: {}'.format(images.shape, images.dtype, labels.dtype))
        
        coord.request_stop()
        coord.join(threads)

        show_image(images.astype(np.uint8), 'image')


def test_tfrecords_with_batch(tfrecord_file, batch_size, num_classes, mode='tf'):
    """test function
    """
    # batch_size = FLAGS.batch_size
    # num_classes = FLAGS.num_classes
    tf_images, tf_labels = read_tfrecords(tfrecord_file)
    tf_images = preprocessing_image(tf_images, mode)
    images_batch, labels_batch = generate_image_label_batch(tf_images, tf_labels, batch_size=batch_size, num_classes=num_classes)
    # open a session 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(batch_size):
            images, labels = sess.run([images_batch, labels_batch])
            print('image shape: {}, image tpye: {}, label type: {}'.format(images.shape, images.dtype, labels.dtype))
        
        coord.request_stop()
        coord.join(threads)

        show_image((images[0, :, :, :]), 'image')
        # print(images[0, :, :, :], labels[0])
# test_tfrecords_with_batch('./17_flowers/tfrecords/valid_tfrecords.tfrecords')

if __name__ == '__main__':
    mode = 'tf'
    batch_size = FLAGS.batch_size
    num_classes = FLAGS.num_classes
    tfrecord_file = FLAGS.tfrecord_file
    print('Test read_tfrecords and visualize image without using batch size')
    test_tfrecords(tfrecords_file)
    
    print('Test read_tfrecords and visualize image using batch size')
    test_tfrecords_with_batch(tfrecords_file, batch_size, num_classes, mode)
    
    
    
