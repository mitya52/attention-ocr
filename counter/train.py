import tensorflow as tf
import numpy as np

from counter.generator import BatchGenerator
from counter.model import Model

from typing import Tuple


def train(epochs: int,
          steps: int,
          batch_size: int,
          image_size: Tuple[int, int],
          grid_size: Tuple[int, int]):
    max_sequence_length = grid_size[0] * grid_size[1]
    images_input = tf.placeholder(shape=(batch_size, image_size[0], image_size[1], 1), dtype=tf.float32)
    sequences_input = tf.placeholder(shape=(batch_size, max_sequence_length), dtype=tf.int32)
    is_training = tf.placeholder(dtype=tf.bool)

    model = Model(
        images_input=images_input,
        sequences_input=sequences_input,
        is_training=is_training,
        max_sequence_length=max_sequence_length)

    train_op = tf.contrib.layers.optimize_loss(
        model.loss,
        tf.train.get_global_step(),
        optimizer='Adam',
        learning_rate=0.001,
        summaries=['loss'])

    generator = BatchGenerator(
        batch_size=batch_size,
        grid_size=grid_size,
        image_size=image_size)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for e in range(epochs):
            for step, (images, sequences) in enumerate(generator):
                sess.run([train_op], feed_dict={
                    images_input: images,
                    sequences_input: sequences,
                    is_training: True})
                if step == steps:
                    break

            images, sequences = next(generator)
            predictions, losses = sess.run([model.predictions, model.loss], feed_dict={
                images_input: images,
                sequences_input: sequences,
                is_training: False})
            print('Epoch {} loss: {}'.format(e, np.mean(losses)))
            print('Sequence:   {}'.format(sequences[0]))
            print('Prediction: {}'.format(predictions[0]))


if __name__ == '__main__':
    epochs = 100
    steps = 100
    batch_size = 64

    image_size = 100, 100
    grid_size = 5, 5

    train(epochs, steps, batch_size, image_size, grid_size)
