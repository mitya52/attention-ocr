#!/usr/bin/python3
import tensorflow as tf
import numpy as np

from generator import Generator
from model import Model

from imgaug import augmenters as iaa

class BatchGenerator(Generator):

	def __init__(self, size, alphabet, end_sym, max_sequence_length, max_lines, batch_size):
		self.batch_size = batch_size
		super(BatchGenerator, self).__init__(size, alphabet, end_sym, max_sequence_length, max_lines)

	def generate_batch(self):
		'''
			Generator of batches (X, y) where:
				X is np.array of size [batch_size, img_h, img_w, 1] and dtype tf.float32
				y is np.array of size [batch_size, max_sequence_length] and dtype tf.int32
		'''

		w, h = self.size
		sl = self.max_sequence_length
		bs = self.batch_size

		X = np.zeros((bs, h, w), dtype=np.uint8)
		y = np.zeros((bs, sl), dtype=np.int32)

		while True:
			length = np.random.randint(1, self.max_sequence_length)
			for i in range(bs):
				X[i], y[i] = self.generate(length)
			yield np.expand_dims(np.float32(X / 256.), axis=3), y

def train(epochs, steps, batch_size, image_size, alphabet, max_sequence_length):
	import time

	img_w, img_h = image_size

	images_input = tf.placeholder(shape=(batch_size, img_h, img_w, 1), dtype=tf.float32)
	sequences_input = tf.placeholder(shape=(batch_size, max_sequence_length), dtype=tf.int32)
	
	model = Model(images_input, sequences_input, max_sequence_length, alphabet)
	endpoints = model.endpoints()

	train_op = tf.contrib.layers.optimize_loss(
			endpoints['loss'],
			tf.train.get_global_step(),
			optimizer='Adam',
			learning_rate=0.001,
			summaries=['loss', 'learning_rate'])
	tf.summary.image('input_images', images_input)
	tf.summary.image('alignments', endpoints['alignments'])
	merged = tf.summary.merge_all()

	generator = BatchGenerator(
		size=image_size,
		alphabet=alphabet,
		end_sym=model.end_symbol(),
		max_sequence_length=max_sequence_length,
		max_lines=3,
		batch_size=batch_size)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)

		def random_name():
			return ''.join([np.random.choice(list('0123456789')) for _ in range(8)])

		train_writer = tf.summary.FileWriter('logs/{}'.format(random_name()), sess.graph)

		ckpt = tf.train.get_checkpoint_state('train/')
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)

		for e in range(epochs):
			t = time.time()
			for step, (imgs, seqs) in enumerate(generator.generate_batch()):
				feed_dict = {
					images_input: imgs,
					sequences_input: seqs,
				}
				if step < steps:
					summary, _ = sess.run([merged, train_op], feed_dict=feed_dict)
				else:
					predictions = sess.run([endpoints['predictions']], feed_dict=feed_dict)
					break

			train_writer.add_summary(summary, e)
			print("Epoch {} ends with time {:.4f}".format(e, time.time() - t))
			print("Predictions: {}".format(predictions))

		saver.save(sess, 'train/model', global_step=epochs)

if __name__ == '__main__':

	image_size = (100, 100)
	
	alphabet = sorted(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
	max_sequence_length = 9
	
	epochs = 100
	steps = 100
	batch_size = 32

	train(epochs, steps, batch_size, image_size, alphabet, max_sequence_length)