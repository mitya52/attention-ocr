#!/usr/bin/python3
import tensorflow as tf
import numpy as np

from generator import BatchGenerator, PrecomputeBatchGenerator
from model import Model

from imgaug import augmenters as iaa

def get_number_parameters(variables):
	total_parameters = 0
	for variable in variables:
		shape = variable.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
		total_parameters += variable_parameters
	return total_parameters

def train(epochs, steps, batch_size, image_size, alphabet, max_sequence_length):
	import time

	img_w, img_h = image_size

	images_input = tf.placeholder(shape=(batch_size, img_h, img_w, 1), dtype=tf.float32)
	sequences_input = tf.placeholder(shape=(batch_size, max_sequence_length), dtype=tf.int32)
	is_training = tf.placeholder(shape=(), dtype=tf.bool)

	model = Model(
		images_input,
		sequences_input,
		is_training,
		max_sequence_length,
		alphabet)
	endpoints = model.endpoints()

	trainable = get_number_parameters(tf.trainable_variables())
	print('Model has {} trainable parameters'.format(trainable))

	train_op = tf.contrib.layers.optimize_loss(
			endpoints['loss'],
			tf.train.get_global_step(),
			optimizer='Adam',
			learning_rate=0.0001,
			summaries=['loss', 'learning_rate'])
	tf.summary.image('input_images', images_input)
	tf.summary.image('alignments', endpoints['alignments'])
	merged = tf.summary.merge_all()

	train_generator = PrecomputeBatchGenerator(
		size=image_size,
		alphabet=alphabet,
		max_sequence_length=max_sequence_length,
		max_lines=3,
		batch_size=batch_size)

	val_generator = PrecomputeBatchGenerator(
		size=image_size,
		alphabet=alphabet,
		max_sequence_length=max_sequence_length,
		max_lines=3,
		batch_size=batch_size,
		precompute_size=1000)

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
			for step, (imgs, seqs) in enumerate(train_generator.generate_batch()):
				if step < steps:
					summary, _ = sess.run([merged, train_op], feed_dict={images_input: imgs, sequences_input: seqs, is_training: True})
				else:
					break
			for imgs, seqs in val_generator.generate_batch():
				predictions = sess.run([endpoints['predictions']], feed_dict={images_input: imgs, is_training: False})[0]
				sequences = seqs
				break

			train_writer.add_summary(summary, e)
			print("Epoch {} ends with time {:.4f}".format(e, time.time() - t))
			print("Expectation: {}".format(sequences[0]))
			print("Reality: {}".format(predictions[0]))

		saver.save(sess, 'train/model', global_step=epochs)

if __name__ == '__main__':

	image_size = network['image_size']
	alphabet = network['alphabet']
	max_sequence_length = network['max_sequence_length']

	epochs = network['epochs']
	steps = network['steps']
	batch_size = network['batch_size']

	train(epochs, steps, batch_size, image_size, alphabet, max_sequence_length)