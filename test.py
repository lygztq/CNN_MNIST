import tensorflow as tf
import numpy as np
import glob, os
from scipy import misc

from model import cnn_model
from utils import dataset
import hyper
IMG_PATH = hyper.TEST_IMG_PATH
LABEL_PATH = hyper.TEST_LABEL_PATH
CKPT_DIR = hyper.CKPT_DIR
IMG_SIZE = hyper.IMG_SIZE


if __name__ == '__main__':
	with tf.Session() as sess:
		input_tensor = tf.placeholder(tf.float32, shape=(None,IMG_SIZE[0],IMG_SIZE[1],1))
		input_label = tf.placeholder(tf.float32, shape=(None,10))
		shared_model = tf.make_template("shared_model", cnn_model)
		output_tensor, weights = shared_model(input_tensor)
		correct_prediction = tf.equal(tf.argmax(output_tensor,1), tf.argmax(input_label,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		init = tf.global_variables_initializer()
		sess.run(init)

		# restore should behind the initialization, otherwise the model value will be rushed.
		ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
		saver = tf.train.Saver(weights)
		saver.restore(sess, tf.train.latest_checkpoint(CKPT_DIR))

		test_set = dataset(IMG_PATH,LABEL_PATH)
		print "use test set: ", IMG_PATH 
		a = sess.run(accuracy,feed_dict={input_tensor:test_set.imgs,input_label:test_set.one_hot_labels()})
		print "The accuracy on test set is: %.4f" % a
