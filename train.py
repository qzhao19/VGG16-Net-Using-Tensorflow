import os
import numpy as np
import tensorflow as tf
from model import vgg16
from evals import calc_loss_acc, train_op
from utils.processing_image import preprocessing_image
from utils.tfrecords import generate_image_label_batch

flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_string('f', '', 'kernel')

flags.DEFINE_integer('valid_steps', 11, 'The number of validation steps ')

flags.DEFINE_integer('max_steps', 1001, 'The number of maximum steps for traing')

flags.DEFINE_integer('batch_size', 128, 'The number of images in each batch during training')

flags.DEFINE_float('base_learning_rate', 0.0001, "base learning rate for optimizer")


flags.DEFINE_integer('num_classes', 20, 'The number of label classes')

flags.DEFINE_float('keep_prob', 0.75, "the probability of keeping neuron unit")


flags.DEFINE_integer('image_height', 64, 'The image height')

flags.DEFINE_integer('image_width', 64, 'The image width')

flags.DEFINE_integer('image_channel', 3, 'The image channels')


flags.DEFINE_string('mode', 'tf', 'Image preprocessing mode')

flags.DEFINE_string('train_tfrecords_file', './17_flowers/tfrecords/train_tfrecords.tfrecords', 'The path points to train tfrecords file')

flags.DEFINE_string('valid_tfrecords_file', './17_flowers/tfrecords/valid_tfrecords.tfrecords', 'The path points to validation tfrecords file')

flags.DEFINE_string('tensorboard_dir', './17_flowers/tensorboard_logs/', 'The path points to tensorboard logs ')

flags.DEFINE_string('saver_dir', './17_flowers/vgg16_model/', 'The path to saved checkpoints')


FLAGS = flags.FLAGS




def train(FLAGS):
    """training model

    """
    batch_size =  FLAGS.batch_size
    num_classes =  FLAGS.num_classes
    
    train_tfrecords_file = FLAGS.train_tfrecords_file
    valid_tfrecords_file = FLAGS.valid_tfrecords_file
    
    tensorboard_dir = FLAGS.tensorboard_dir
    
    saver_dir = FLAGS.saver_dir

    image_height, image_width, image_channel = (FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel)

    mode = FLAGS.mode

    base_learning_rate = FLAGS.base_learning_rate

    max_steps = FLAGS.max_steps

    valid_steps = FLAGS.valid_steps

    keep_prob = FLAGS.keep_prob

    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []


    with tf.Graph().as_default():
        # define image and label placehold
        images_pl = tf.placeholder(tf.float32, shape=[batch_size, image_height, image_width, image_channel])
        labels_pl = tf.placeholder(tf.int32, shape=[batch_size, num_classes])
        is_training_pl = tf.placeholder(tf.bool, name='is_training')
        # keep_prob_pl = tf.placeholder(tf.float32)

        # get training image with batch 
        train_tf_image, train_tf_label = read_tfrecords(train_tfrecords_file, shape=(image_height, image_width, image_channel))
        train_tf_image = preprocessing_image(train_tf_image, mode=mode)
        train_tf_image_batch, train_tf_label_batch = generate_image_label_batch(train_tf_image, train_tf_label, batch_size, num_classes)

        # validation image with batch
        valid_tf_image, valid_tf_label = read_tfrecords(valid_tfrecords_file, shape=(image_height, image_width, image_channel))
        valid_tf_image = preprocessing_image(valid_tf_image, mode=mode)
        valid_tf_image_batch, valid_tf_label_batch = generate_image_label_batch(valid_tf_image, valid_tf_label, batch_size, num_classes)

        # compute logits from model
        logits = vgg16(images_pl, num_classes, keep_prob, is_training_pl)
        # get global steps
        global_steps = tf.Variable(0, trainable=False)
        # compute loss, accucracy and predictions
        loss, accuracy, _ = calc_loss_acc(labels_pl, logits)
        # training operator to update model params
        training_op = train_op(loss, global_steps, base_learning_rate, option='SGD')

        # define the model saver
        saver = tf.train.Saver(tf.global_variables())
        
        # define a summary operation 
        summary_op = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            # start queue runner
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            # set up summary file writer
            train_writer = tf.summary.FileWriter(tensorboard_dir + 'train', sess.graph)
            valid_writer = tf.summary.FileWriter(tensorboard_dir + 'valid')

            for step in range(max_steps):
                # load image and label with batch size
                train_image_batch, train_label_batch = sess.run([train_tf_image_batch, train_tf_label_batch])
                # define training feed dict
                train_feed_dict = {images_pl: train_image_batch, labels_pl: train_label_batch, is_training_pl: True}
                _, _loss, _acc, _summary_op = sess.run([training_op, loss, accuracy, summary_op], feed_dict=train_feed_dict)

                # store loss and accuracy value
                train_loss.append(_loss)
                train_acc.append(_acc)

                # print loss and acc
                # print("Iteration " + str(step) + ", Mini-batch Loss= " + "{:.6f}".format(_loss) + ", Training Accuracy= " + "{:.5f}".format(_acc))

                # if step%10 == 0:
                    # _logits = sess.run(logits, feed_dict=train_feed_dict)
                    # print('Per class accuracy by logits in training time', per_class_acc(train_label_batch, _logits))
                print("Iteration " + str(step) + ", Mini-batch Loss= " + "{:.6f}".format(_loss) + ", Training Accuracy= " + "{:.5f}".format(_acc))
                train_writer.add_summary(_summary_op, step)
                train_writer.flush()

                if step%5 == 0:
                    print('start validation process')
                    _valid_loss, _valid_acc = [], []

                    for valid_step in range(valid_steps):
                        valid_image_batch, valid_label_batch = sess.run([valid_tf_image_batch, valid_tf_label_batch])

                        valid_feed_dict = {images_pl: valid_image_batch, labels_pl: valid_label_batch, is_training_pl: True}

                        _loss, _acc, _summary_op = sess.run([loss, accuracy, summary_op], feed_dict = valid_feed_dict)

                        valid_writer.add_summary(_summary_op, valid_step)
                        valid_writer.flush()

                        _valid_loss.append(_loss)
                        _valid_acc.append(_acc)

                    valid_loss.append(np.mean(_valid_loss))
                    valid_acc.append(np.mean(_valid_acc))
                    print("Iteration {}: Train Loss {:6.3f}, Train Acc {:6.3f}, Val Loss {:6.3f}, Val Acc {:6.3f}".format(step, train_loss[-1], train_acc[-1], valid_loss[-1], valid_acc[-1]))

            checkpoint_path = os.path.join(saver_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
                
            coord.request_stop()
            coord.join(threads)
