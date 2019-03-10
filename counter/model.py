from tensorflow.contrib import slim, seq2seq, layers, rnn
from tensorflow.python.ops import array_ops

import tensorflow as tf
import numpy as np
from typing import Optional, List

__all__ = ['Model']


class ConvTower:
    def __init__(self,
                 inputs: tf.Tensor,
                 is_training: tf.Tensor):
        self.inputs = inputs
        self.is_training = is_training

    def _conv_block(self,
                    inputs: tf.Tensor,
                    filters: int) -> tf.Tensor:
        conv = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)
        return tf.layers.dropout(inputs=pool, rate=0.2, training=self.is_training)

    def create(self,
               blocks: List[int],
               output_filters: Optional[int]) -> tf.Tensor:
        x = self.inputs
        for filters in blocks:
            x = self._conv_block(x, filters)

        # 1x1 convolution for dimension reduction
        if output_filters:
            x = tf.layers.conv2d(
                inputs=x,
                filters=output_filters,
                kernel_size=[1, 1],
                padding="same",
                activation=tf.nn.relu)

        return x


class Attention:
    def __init__(self,
                 feature_map: tf.Tensor,
                 sequences: tf.Tensor,
                 max_sequence_length: int):
        self.feature_map = feature_map
        self.sequences = sequences
        self.max_sequence_length = max_sequence_length

        self.num_units = 32
        self.alphabet_size = 1
        self.complete_alphabet_size = self.alphabet_size + 2

        # special symbol codes
        self.eos = self.alphabet_size
        self.go = self.alphabet_size + 1

        self.batch_size = array_ops.shape(self.feature_map)[0]
        self.start_tokens = tf.ones([self.batch_size], dtype=tf.int32) * self.go
        self.lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(sequences, self.eos)), axis=1) + 1

    def create_train_branch(self):
        # prepare sequences
        sequences = tf.concat([tf.expand_dims(self.start_tokens, 1), self.sequences], axis=1)
        sequences = slim.one_hot_encoding(sequences, num_classes=self.complete_alphabet_size)

        helper = seq2seq.TrainingHelper(
            inputs=sequences,
            sequence_length=self.lengths)

        outputs = self._decode(helper=helper, reuse=False)

        # complete logits to [batch_size, max_sequence_length, complete_alphabet_size]
        logits = outputs[0].rnn_output
        to_complete = self.max_sequence_length - array_ops.shape(logits)[1]

        def complete_with_eos(logits, to_complete):
            zeros_first = tf.zeros((self.batch_size, to_complete, self.alphabet_size), dtype=tf.float32)
            zeros_last = tf.zeros((self.batch_size, to_complete, 1), dtype=tf.float32)
            ones = tf.ones((self.batch_size, to_complete, 1), dtype=tf.float32)
            completion = tf.concat([zeros_first, ones, zeros_last], axis=2)
            return tf.concat([logits, completion], axis=1)

        logits = tf.cond(
            tf.equal(to_complete, 0),
            lambda: logits,
            lambda: complete_with_eos(logits, to_complete))

        return logits

    def create_predict_branch(self):
        helper = seq2seq.GreedyEmbeddingHelper(
            embedding=lambda x: layers.one_hot_encoding(x, self.complete_alphabet_size),
            start_tokens=self.start_tokens,
            end_token=self.eos)

        outputs = self._decode(helper=helper, reuse=True)

        return tf.argmax(tf.nn.sigmoid(outputs[0].rnn_output), axis=2)

    def _decode(self,
                helper: seq2seq.Helper,
                reuse: bool = False):
        with tf.variable_scope('decoder', reuse=reuse):
            attention_mechanism = seq2seq.BahdanauAttention(
                num_units=self.num_units,
                memory=self.feature_map)
            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(
                num_units=self.num_units)
            attention_cell = seq2seq.AttentionWrapper(
                cell=rnn_cell,
                attention_mechanism=attention_mechanism,
                alignment_history=True)

            output_cell = rnn.OutputProjectionWrapper(
                cell=attention_cell,
                output_size=self.complete_alphabet_size,
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


class Model:
    def __init__(self,
                 images_input: tf.placeholder,
                 sequences_input: tf.placeholder,
                 is_training: tf.placeholder,
                 max_sequence_length: int):
        conv_tower_model = ConvTower(
            inputs=images_input,
            is_training=is_training)

        # convolution tower
        x = conv_tower_model.create(
            blocks=[32, 64, 128],
            output_filters=32)

        # preprocess for attention model
        x = Model._encode_coordinates(x)
        ct_out_shape = array_ops.shape(x)
        x = tf.reshape(x, [ct_out_shape[0], -1, ct_out_shape[3]])

        attention_model = Attention(
            feature_map=x,
            sequences=sequences_input,
            max_sequence_length=max_sequence_length)

        self.loss = self._create_loss(
            sequences_input=sequences_input,
            attention_model=attention_model)
        self.predictions = attention_model.create_predict_branch()

    @staticmethod
    def _encode_coordinates(feature_map):
        batch_size, h, w, _ = feature_map.shape.as_list()
        x, y = tf.meshgrid(tf.range(w), tf.range(h))
        w_loc = slim.one_hot_encoding(x, num_classes=w)
        h_loc = slim.one_hot_encoding(y, num_classes=h)
        loc = tf.concat([h_loc, w_loc], 2)
        loc = tf.tile(tf.expand_dims(loc, 0), [batch_size, 1, 1, 1])
        return tf.concat([feature_map, loc], 3)

    @staticmethod
    def _create_loss(sequences_input: tf.Tensor,
                     attention_model: Attention) -> tf.Tensor:
        logits = attention_model.create_train_branch()
        weights = tf.sequence_mask(attention_model.lengths, attention_model.max_sequence_length)
        return seq2seq.sequence_loss(
            logits=logits,
            targets=sequences_input,
            weights=tf.cast(weights, dtype=tf.float32))


if __name__ == '__main__':
    batch_size = 32
    image_shape = 100, 100, 1
    max_sequence_length = 10

    images_input = tf.placeholder(shape=(batch_size,) + image_shape, dtype=tf.float32)
    sequences_input = tf.placeholder(shape=(batch_size, max_sequence_length), dtype=tf.int32)
    is_training = tf.placeholder(dtype=tf.bool)

    model = Model(
        images_input=images_input,
        sequences_input=sequences_input,
        is_training=is_training,
        max_sequence_length=max_sequence_length)

    # check train
    train_op = layers.optimize_loss(
        model.loss,
        tf.train.get_global_step(),
        optimizer='Adam',
        learning_rate=0.001)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        images = np.ones((batch_size,) + image_shape, dtype=np.float32)
        sequences = np.zeros((batch_size, max_sequence_length), dtype=np.float32)
        sequences[:, max_sequence_length // 2:] = 1
        sess.run([train_op], feed_dict={
            images_input: images,
            sequences_input: sequences,
            is_training: True})

    # check test
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        images = np.ones((batch_size,) + image_shape, dtype=np.float32)
        sequences[:, max_sequence_length // 2:] = 1
        fetched = sess.run([model.predictions], feed_dict={
            images_input: images,
            is_training: False})