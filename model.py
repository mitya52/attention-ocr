#!/usr/bin/python3
from tensorflow.contrib import slim
from tensorflow.python.ops import array_ops

import tensorflow as tf
import numpy as np

class Model():

	def __init__(self, images_input, sequences_input, max_sequence_length, alphabet):
		# we add 1 because of start symbol
		self.max_sequence_length = max_sequence_length + 1
		self.alphabet_size = len(alphabet) + 1

		features = self._conv_tower(images_input)
		features = self._encode_coordinates(features)
		features_h = array_ops.shape(features)[1]
		features = self._flatten(features)

		sequences_input, self.start_tokens = self._add_start_tokens(sequences_input, len(alphabet))
		sequences, lengths, weights = self._prepare_sequences(sequences_input)
		logits, self.alignments, self.predictions = self._create_attention(features, sequences, lengths, features_h)
		self.loss = self._create_loss(logits, sequences_input, weights)

	def end_symbol(self):
		return self.alphabet_size + 1

	def endpoints(self):
		return {
			"loss": self.loss,
			"alignments": self.alignments,
			"predictions": self.predictions
		}

	def _add_start_tokens(self, seq, start_sym):
		batch_size = seq.shape.as_list()[0]
		start_tokens = tf.ones([batch_size], dtype=tf.int32)*start_sym
		return tf.concat([tf.expand_dims(start_tokens, 1), seq], axis=1), start_tokens

	def _prepare_sequences(self, seq):
		lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(seq, self.alphabet_size)), axis=1)
		sequences = slim.one_hot_encoding(seq, num_classes=self.alphabet_size+1)
		weights = tf.cast(tf.sequence_mask(lengths, self.max_sequence_length), dtype=tf.float32)
		return sequences, lengths, weights

	def _conv_tower(self, inp):

		def conv_block(inp, filters):
			conv = tf.layers.conv2d(
				inputs=inp,
				filters=filters,
				kernel_size=[3, 3],
				padding="same",
				activation=tf.nn.relu)
			pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)
			return tf.layers.dropout(inputs=pool, rate=0.9, training=True) # TODO

		features = conv_block(inp, 16)
		features = conv_block(features, 32)
		features = conv_block(features, 64)

		return features

	def _encode_coordinates(self, features):
		batch_size, h, w, _ = features.shape.as_list()
		x, y = tf.meshgrid(tf.range(w), tf.range(h))
		w_loc = slim.one_hot_encoding(x, num_classes=w)
		h_loc = slim.one_hot_encoding(y, num_classes=h)
		loc = tf.concat([h_loc, w_loc], 2)
		loc = tf.tile(tf.expand_dims(loc, 0), [batch_size, 1, 1, 1])
		return tf.concat([features, loc], 3)	

	def _flatten(self, features):
		batch_size = features.get_shape().dims[0].value
		feature_size = features.get_shape().dims[3].value
		return tf.reshape(features, [batch_size, -1, feature_size])

	def _create_attention(self, memory, sequences, lengths, features_h, num_units=32):
		batch_size = memory.shape.as_list()[0]

		train_helper = tf.contrib.seq2seq.TrainingHelper(
			sequences,
			lengths)

		embeddings = lambda x: tf.contrib.layers.one_hot_encoding(x, self.alphabet_size+1)
		pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
			embeddings,
			start_tokens=self.start_tokens,
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
					self.alphabet_size+1,
					reuse=reuse)
				decoder = tf.contrib.seq2seq.BasicDecoder(
					cell=output_cell,
					helper=helper,
					initial_state=output_cell.zero_state(
						dtype=tf.float32,
						batch_size=batch_size))
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
			alignments = tf.reshape(alignments, [batch_size, features_h, -1])
			return tf.expand_dims(alignments, axis=3)

		alignments = alignments_mean(train_outputs[1].alignment_history.stack())

		# complete logits to [batch_size, max_sequence_length, alphabet_size]
		logits = train_outputs[0].rnn_output
		to_complete = self.max_sequence_length - array_ops.shape(logits)[1]

		def complete(logits, to_complete):
			zeros = tf.zeros((batch_size, to_complete, self.alphabet_size), dtype=tf.float32)
			ones = tf.ones((batch_size, to_complete, 1), dtype=tf.float32)
			completion = tf.concat([zeros, ones], axis=2)
			return tf.concat([logits, completion], axis=1)

		logits = tf.cond(
			tf.equal(to_complete, 0),
			lambda: logits,
			lambda: complete(logits, to_complete))

		return logits, alignments, pred_outputs[0].sample_id

	def _create_loss(self, logits, sequences, weights):
		return tf.contrib.seq2seq.sequence_loss(
			logits,
			sequences,
			weights)

	def _create_preds(self, logits, sequences_input):
		preds = tf.argmax(tf.nn.softmax(logits), axis=2, output_type=tf.int32)
		preds = tf.expand_dims(preds, axis=2)
		sequences_input = tf.expand_dims(sequences_input, axis=2)
		preds = tf.concat([preds, sequences_input], axis=2)
		return preds

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
