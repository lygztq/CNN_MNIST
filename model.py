import tensorflow as tf
import numpy as np

import hyper
IMG_SIZE = hyper.IMG_SIZE

def PReLU(input_x, index):
	alphas = tf.get_variable(
		'alpha_%02d' % index,
		input_x.get_shape()[-1],
		initializer=tf.constant_initializer(1e-4),
		dtype=tf.float32
	)
	return tf.maximum(0.0, input_x) + alphas * tf.minimum(0.0,input_x)

def cnn_model(input_tensor):
	with tf.device("/gpu:0"):
		weights = []
		tensor = None

		# First layer
		conv_00_w = tf.get_variable("conv_00_w", [3,3,1,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9)))
		conv_00_b = tf.get_variable("conv_00_b", [64], initializer=tf.constant_initializer(0))
		weights.append(conv_00_w)
		weights.append(conv_00_b)
		tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1,1,1,1], padding='SAME'), conv_00_b))
		#tensor = PReLU(tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1,1,1,1], padding='SAME'), conv_00_b),0)

		# Middle layer
		for i in range(2):
			#conv_w = tf.get_variable("conv_%02d_w" % (i+1), [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer())
			conv_w = tf.get_variable("conv_%02d_w" % (i+1), [3,3,64,64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/64)))
			conv_b = tf.get_variable("conv_%02d_b" % (i+1), [64], initializer=tf.constant_initializer(0))
			weights.append(conv_w)
			weights.append(conv_b)
			tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b))
			#tensor = PReLU(tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,1,1,1], padding='SAME'), conv_b), i+1)

		# Last layer
		fc_w = tf.get_variable("fc_00_w", [IMG_SIZE[0] * IMG_SIZE[1] * 64, 512], initializer=tf.random_normal_initializer(stddev=0.1))
		fc_b = tf.get_variable("fc_00_b", [512], initializer=tf.constant_initializer(0))
		tensor = tf.nn.relu(tf.matmul(tf.reshape(tensor,[-1,IMG_SIZE[0]*IMG_SIZE[1]*64]),fc_w) + fc_b)
		weights.append(fc_w)
		weights.append(fc_b)

		# dropout
		#keep_prob = tf.placeholder("float")
		#tensor = tf.nn.dropout(tensor, keep_prob)

		# softmax
		sf_w = tf.get_variable("sf_00_w",[512, 10], initializer=tf.random_normal_initializer(stddev=0.1))
		sf_b = tf.get_variable("sf_00_b",[10], initializer=tf.constant_initializer(0))
		tensor = tf.nn.softmax(tf.matmul(tensor,sf_w) + sf_b,name='output_tensor')
		weights.append(sf_w)
		weights.append(sf_b)

		return tensor, weights
