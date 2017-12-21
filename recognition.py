import tensorflow as tf
import numpy as np
import glob, os, argparse, PIL 
from scipy import misc
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

import hyper
from model import cnn_model
CKPT_DIR = hyper.CKPT_DIR
IMG_SIZE = hyper.IMG_SIZE

parser = argparse.ArgumentParser()
parser.add_argument("--img")
args = parser.parse_args()
img_path = args.img

def change_format(img):
	"""
	Change scipy img format into MNIST format
	"""
	img = 255-img
	img = img.astype(np.float)/255
	return img

def recognition_img(sess,img):
	#saver.restore(sess, tf.train.latest_checkpoint(CKPT_DIR))
	answer = tf.argmax(output_tensor,1)
	a = sess.run(answer, feed_dict={input_tensor:img})
	return a

if __name__ == '__main__':
	raw_img = misc.imread(img_path,'L')
	img_to_recog = misc.imresize(raw_img,(IMG_SIZE[0],IMG_SIZE[1]))
	img_to_recog = change_format(img_to_recog)
	img_to_recog = np.reshape([img_to_recog], [1, IMG_SIZE[0], IMG_SIZE[1], 1])

	with tf.Session() as sess:
		input_tensor = tf.placeholder(tf.float32, shape=(1,IMG_SIZE[0],IMG_SIZE[1],1))
		shared_model = tf.make_template('shared_model',cnn_model)
		output_tensor, weights = shared_model(input_tensor)

		init = tf.global_variables_initializer()
		sess.run(init)

		ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
		saver = tf.train.Saver(weights)
		saver.restore(sess, tf.train.latest_checkpoint(CKPT_DIR))

		print "The number is: ", recognition_img(sess, img_to_recog)

