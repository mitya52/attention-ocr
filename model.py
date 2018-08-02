#!/usr/bin/python3
from tensorflow.contrib import slim
from tensorflow.python.ops import array_ops

import tensorflow as tf
import numpy as np

class Model():

	def __init__(self, images_input, sequences_input, is_training, max_sequence_length, alphabet, alignments_type='mean'):
		self.max_sequence_length = max_sequence_length
		self.alphabet_size = len(alphabet)
		self.batch_size = images_input.shape.as_list()[0]

		features = self._conv_tower(images_input, is_training)
		features = self._encode_coordinates(features)
		features_h = array_ops.shape(features)[1]
		features = self._flatten(features)

		sequences, start_tokens, lengths, weights = self._prepare_sequences(
			seq=sequences_input)
		logits, self.alignments, self.predictions = self._create_attention(
			memory=features,
			sequences=sequences,
			lengths=lengths,
			features_h=features_h,
			start_tokens=start_tokens,
			alignments_type=alignments_type)
		self.lengths = lengths
		self.weights = weights
		self.train_predictions = self._create_train_predictions(logits)
		self.loss = self._create_loss(logits, sequences_input, weights)

	def endpoints(self):
		return {
			"loss": self.loss,
			"lengths": self.lengths,
			"weights": self.weights,
			"train_predictions": self.train_predictions,
			"alignments": self.alignments,
			"predictions": self.predictions,
		}

	def _conv_tower(self, inp, is_training):

		def conv_block(inp, filters):
			conv = tf.layers.conv2d(
				inputs=inp,
				filters=filters,
				kernel_size=[3, 3],
				padding="same",
				activation=tf.nn.relu)
			pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)
			return tf.layers.dropout(inputs=pool, rate=0.2, training=is_training)

		features = conv_block(inp, 32)
		features = conv_block(features, 64)
		features = conv_block(features, 128)

		# 1x1 convolution for dimention reduction
		features = tf.layers.conv2d(
				inputs=features,
				filters=32,
				kernel_size=[1, 1],
				padding="same",
				activation=tf.nn.relu)

		return features

	def _encode_coordinates(self, features):
		_, h, w, _ = features.shape.as_list()
		x, y = tf.meshgrid(tf.range(w), tf.range(h))
		w_loc = slim.one_hot_encoding(x, num_classes=w)
		h_loc = slim.one_hot_encoding(y, num_classes=h)
		loc = tf.concat([h_loc, w_loc], 2)
		loc = tf.tile(tf.expand_dims(loc, 0), [self.batch_size, 1, 1, 1])
		return tf.concat([features, loc], 3)	

	def _flatten(self, features):
		feature_size = features.get_shape().dims[3].value
		return tf.reshape(features, [self.batch_size, -1, feature_size])

	def _add_start_tokens(self, seq, start_sym):
		start_tokens = tf.ones([self.batch_size], dtype=tf.int32)*start_sym
		return tf.concat([tf.expand_dims(start_tokens, 1), seq], axis=1), start_tokens

	def _prepare_sequences(self, seq):
		lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(seq, self.alphabet_size)), axis=1)
		weights = tf.cast(tf.sequence_mask(lengths, self.max_sequence_length+1), dtype=tf.float32)
		#weights = tf.ones((self.batch_size, self.max_sequence_length+1), dtype=tf.float32)
		seq_train, start_tokens = self._add_start_tokens(
			seq=seq, start_sym=self.alphabet_size+1)
		sequences = slim.one_hot_encoding(seq_train, num_classes=self.alphabet_size+2)
		return sequences, start_tokens, lengths, weights

	def _create_attention(self, memory, sequences, lengths, features_h, start_tokens, alignments_type, num_units=32):
		train_helper = tf.contrib.seq2seq.TrainingHelper(
			inputs=sequences,
			sequence_length=lengths)

		embeddings = lambda x: tf.contrib.layers.one_hot_encoding(x, self.alphabet_size+2)
		pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
			embeddings,
			start_tokens=start_tokens,
			end_token=self.alphabet_size)

		def decode(helper, reuse=False):
			with tf.variable_scope('decode', reuse=reuse):
				attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
					num_units,
					memory)
				cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
				attention_cell = tf.contrib.seq2seq.AttentionWrapper(
					cell,
					attention_mechanism,
					alignment_history=True,
					name="attention")
				output_cell = tf.contrib.rnn.OutputProjectionWrapper(
					attention_cell,
					self.alphabet_size+2,
					reuse=reuse)
				decoder = tf.contrib.seq2seq.BasicDecoder(
					cell=output_cell,
					helper=helper,
					initial_state=output_cell.zero_state(
						dtype=tf.float32,
						batch_size=self.batch_size))
				outputs = tf.contrib.seq2seq.dynamic_decode(
					decoder=decoder,
					output_time_major=False,
					impute_finished=True,
					maximum_iterations=self.max_sequence_length)
			return outputs

		train_outputs = decode(train_helper)
		pred_outputs = decode(pred_helper, reuse=True)

		def alignments_mean(alignments):
			alignments = tf.reduce_mean(alignments, axis=0, keep_dims=False)
			alignments = tf.reshape(alignments, [self.batch_size, features_h, -1])
			return tf.expand_dims(alignments, axis=3)

		def alignments_full(alignments):
			depth = array_ops.shape(alignments)[0]
			alignments = tf.transpose(alignments, [1, 0, 2])
			alignments = tf.reshape(alignments, [self.batch_size, depth, features_h, -1])
			return alignments

		if alignments_type == 'mean':
			alignments = alignments_mean(train_outputs[1].alignment_history.stack())
		elif alignments_type == 'full':
			alignments = alignments_full(pred_outputs[1].alignment_history.stack())
		else:
			raise ValueError('Unknown alignments type "{}"'.format(alignments_type))

		# complete logits to [batch_size, max_sequence_length+1, alphabet_size]
		logits = train_outputs[0].rnn_output
		to_complete = self.max_sequence_length + 1 - array_ops.shape(logits)[1]

		def complete(logits, to_complete):
			zeros_first = tf.zeros((self.batch_size, to_complete, self.alphabet_size), dtype=tf.float32)
			zeros_last = tf.zeros((self.batch_size, to_complete, 1), dtype=tf.float32)
			ones = tf.ones((self.batch_size, to_complete, 1), dtype=tf.float32)
			completion = tf.concat([zeros_first, ones, zeros_last], axis=2)
			return tf.concat([logits, completion], axis=1)

		logits = tf.cond(
			tf.equal(to_complete, 0),
			lambda: logits,
			lambda: complete(logits, to_complete))

		return logits, alignments, tf.argmax(tf.nn.sigmoid(pred_outputs[0].rnn_output), axis=2)

	def _create_train_predictions(self, logits):
		return tf.argmax(logits, axis=2)

	def _create_loss(self, logits, sequences, weights):
		end_tokens = tf.ones((self.batch_size, 1), dtype=tf.int32) * self.alphabet_size
		targets = tf.concat([sequences, end_tokens], axis=1)
		return tf.contrib.seq2seq.sequence_loss(
			logits=logits,
			targets=targets,
			weights=weights)

if __name__ == '__main__':
	img_w, img_h = 100, 20
	max_sequnce_length = 20
	batch_size = 32
	alphabet = list('0123456789')

	images_input = tf.placeholder(shape=(batch_size, img_h, img_w, 1), dtype=tf.float32)
	sequences_input = tf.placeholder(shape=(batch_size, max_sequnce_length), dtype=tf.int32)
	
	model = Model(images_input, sequences_input, max_sequnce_length, alphabet)
	endpoints = model.endpoints()

	train_op = tf.contrib.layers.optimize_loss(
			endpoints['loss'],
			tf.train.get_global_step(),
			optimizer='Adam',
			learning_rate=0.001,
			summaries=['loss', 'learning_rate'])

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)

		feed_dict = {
			images_input: np.random.random((batch_size, img_h, img_w, 1)),
			sequences_input: np.random.randint(0, len(alphabet), size=(batch_size, max_sequnce_length)),
		}
		fetched = sess.run(train_op, feed_dict=feed_dict)
		print(fetched.shape)
