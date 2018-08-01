#!/usr/bin/python3
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

import cv2
import numpy as np

class Generator():
	'''
		This class generates images of self.size with random
		sequences of variable length (not grteater than
		self.max_sequence_length) composed from self.alphabet
		symbols and located line-by-line with self.max_lines
		limit.
	'''

	def __init__(self, size, alphabet, max_sequence_length, max_lines, end_sym=None):
		self.size = size
		self.alphabet = alphabet
		self.max_sequence_length = max_sequence_length
		self.max_lines = max_lines
		self.line_length = max_sequence_length//max_lines
		if not end_sym:
			self.end_sym = len(alphabet)

		self.font_name = 'font.ttf'
		self.font = ImageFont.truetype(self.font_name, 25)

		# this is monospace font
		self.font_w, self.font_h = self.font.getsize('0')
		self.inner_size = self.line_length*self.font_w, self.max_lines*self.font_h
		margin = 0.3
		self.border = int(self.inner_size[0]*margin), int(self.inner_size[1]*margin)
		self.inner_size = self.inner_size[0] + 2*self.border[0], self.inner_size[1] + 2*self.border[1]

	def _random_sequence(self, length=None):
		if not length:
			length = np.random.randint(1, self.max_sequence_length)
		if length > self.max_sequence_length:
			raise ValueError('length must me less or equal max_sequence_length')
		return ''.join([np.random.choice(list(self.alphabet)) for _ in range(length)])

	def _generate_image(self, text):
		ww, hh = self.inner_size
		img = np.zeros((hh, ww), dtype=np.uint8)

		img = PIL.Image.fromarray(img)
		draw = ImageDraw.Draw(img)

		for line in range(int(np.ceil(len(text)/self.line_length))):
			i = self.line_length * line
			ii = min(len(text), i + self.line_length)
			p = (self.border[0], self.border[1] + line * self.font_h)
			draw.text(p, text[i:ii], (255), font=self.font)
		return np.asarray(img)

	def _to_sequence(self, text, complete=True):
		completion = (self.max_sequence_length - len(text)) * complete
		return [self.alphabet.index(w) for w in text] + [self.end_sym] * completion

	def generate(self, length=None):
		text = self._random_sequence(length)
		img = self._generate_image(text)
		if self.size:
			img = cv2.resize(img, self.size)
		return img, np.array(self._to_sequence(text))


class BatchGenerator(Generator):

	def __init__(self, size, alphabet, max_sequence_length, max_lines, batch_size, end_sym=None):
		self.batch_size = batch_size
		super(BatchGenerator, self).__init__(size, alphabet, max_sequence_length, max_lines, end_sym)

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


if __name__ == '__main__':
	size = (200, 200)
	alphabet = sorted('ABCDEFGHIGKLMNOPQRSTUVWXYZ0123456789')

	g = Generator(
		size=size,
		alphabet=alphabet,
		max_sequence_length=9,
		max_lines=3)

	while True:
		img, seq = g.generate()
		print(seq)
		cv2.imshow('image', img)

		k = cv2.waitKey(0) & 0xFF
		if k == 27:
			break