import tensorflow as tf
import numpy as np
import os

import hyper
from model import cnn_model
from utils import dataset

# Get hyper-parameters
IMG_PATH = hyper.IMG_PATH
LABEL_PATH = hyper.LABEL_PATH
IMG_SIZE = hyper.IMG_SIZE
USE_QUEUE_LOADING = hyper.USE_QUEUE_LOADING
BATCH_SIZE = hyper.BATCH_SIZE
BASE_LR_RATE = hyper.BASE_LR_RATE
DECAY_RATE = hyper.DECAY_RATE
STEP = hyper.STEP
MAX_EPOCH = hyper.MAX_EPOCH
SAVE_EVERY = hyper.SAVE_EVERY
CKPT_DIR = hyper.CKPT_DIR
LOG_DIR = hyper.LOG_DIR

def train():
	# Define input
	train_set = dataset(IMG_PATH,LABEL_PATH)
	train_input = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1],1),name='input_tensor')
	train_label = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 10),name='input_label')

	# Define model
	shared_model = tf.make_template('shared_model', cnn_model)
	train_output, weights = shared_model(train_input)

	loss = -tf.reduce_sum(train_label*tf.log(train_output))
	for w in weights:
		loss += tf.nn.l2_loss(w)*1e-4
	tf.summary.scalar("loss", loss)

	correct_prediction = tf.equal(tf.argmax(train_output,1), tf.argmax(train_label,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name='acc')

	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(BASE_LR_RATE, global_step*BATCH_SIZE, STEP, DECAY_RATE)
	tf.summary.scalar("learning_rate", learning_rate)
		
	optimizer = tf.train.AdamOptimizer(learning_rate)
	opt = optimizer.minimize(loss, global_step=global_step)

	saver = tf.train.Saver(weights, max_to_keep=0) # save weights
	config = tf.ConfigProto(allow_soft_placement=True) # use for session initialization, as configure for session
	
	# Training
	with tf.Session(config=config) as sess:
		# Record the training process using tensorboard
		if not os.path.exists(LOG_DIR):
			os.mkdir(LOG_DIR)
		if not os.path.exists(CKPT_DIR):
			os.mkdir(CKPT_DIR)
		merged = tf.summary.merge_all() # used for tensorboard, to merge all data together
		file_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

		# initialize all variables
		init = tf.global_variables_initializer()
		sess.run(init)

		# Start training
		for epoch in xrange(0, MAX_EPOCH):
			batch_input, batch_label = train_set.next_batch(BATCH_SIZE)
			_, l, output, lr, g_step, summary, acc = sess.run(
				[opt, loss, train_output, learning_rate, global_step, merged, accuracy],
				feed_dict={train_input:batch_input, train_label:batch_label}
				)
			print "[step %05d] loss %.4f\t lr %.6f acc %.4f" % (g_step, np.sum(l)/BATCH_SIZE, lr, acc)
			file_writer.add_summary(summary, epoch)
			
			if (epoch+1) % SAVE_EVERY == 0:
				print "save: ", os.path.join(CKPT_DIR, "CNN_MNIST_%05d.ckpt" % epoch)
				saver.save(sess, os.path.join(CKPT_DIR,"CNN_MNIST_%05d.ckpt" % epoch), global_step=global_step) 

if __name__ == '__main__':
	train()
