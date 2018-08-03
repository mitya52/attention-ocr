#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import cv2

from generator import BatchGenerator
from model import Model
from config import network

def predict(image_size, alphabet, max_sequence_length):
	img_w, img_h = image_size

	images_input = tf.placeholder(shape=(1, img_h, img_w, 1), dtype=tf.float32)
	sequences_input = tf.placeholder(shape=(1, max_sequence_length), dtype=tf.int32)
	is_training = tf.constant(False, dtype=tf.bool)

	model = Model(
		images_input,
		sequences_input,
		is_training,
		max_sequence_length,
		alphabet,
		alignments_type='full')
	endpoints = model.endpoints()

	test_generator = BatchGenerator(
		size=image_size,
		alphabet=alphabet,
		max_sequence_length=max_sequence_length,
		max_lines=3,
		batch_size=1)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)

		ckpt = tf.train.get_checkpoint_state('train/')
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			raise Exception('Cannot load checkpoint')

		for imgs, seqs in test_generator.generate_batch():
			predictions, alignments = sess.run([endpoints['predictions'], endpoints['alignments']], feed_dict={images_input: imgs})
			img = np.squeeze(imgs[0])

			predicted_text = ''.join([alphabet[x] for x in predictions[0] if x < len(alphabet)])
			print("Predicted: {}".format(predicted_text))
			cv2.imshow('original', img)

			font = cv2.FONT_HERSHEY_SIMPLEX
			for ind, alignment in enumerate(alignments[0]):
				h, w = img.shape[:2]
				img_al = cv2.resize(alignment, (w, h), interpolation=cv2.INTER_AREA)
				highlighted = cv2.resize((img + img_al * 2) / 3., (500, 300), interpolation=cv2.INTER_AREA)
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(highlighted, predicted_text[ind], (50, 50), font, 2, (255,255,255), 2, cv2.LINE_AA)
				cv2.imshow('alignment', highlighted)
				k = cv2.waitKey(1000) & 0xFF
				if k == 27:
					break

			k = cv2.waitKey(0) & 0xFF
			if k == 27:
				break

if __name__ == '__main__':

	image_size = network['image_size']
	alphabet = network['alphabet']
	max_sequence_length = network['max_sequence_length']

	predict(image_size, alphabet, max_sequence_length)