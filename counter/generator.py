import cv2
import random
import numpy as np
from copy import deepcopy
from itertools import product
from typing import Optional, Tuple, List


__all__ = ['Generator']


class Generator:
    def __init__(self,
                 grid_size: Tuple[int, int],
                 image_size: Optional[Tuple[int, int]] = None):
        self.grid_size = grid_size
        self.image_size = image_size

        self.inner_cell_size = 50, 50

    def _total_cell_count(self):
        return self.grid_size[0] * self.grid_size[1]

    def _random_sequence(self,
                         length: Optional[int] = None):
        if not length:
            length = np.random.randint(1, self._total_cell_count())
        elif length > self._total_cell_count():
            raise ValueError('length must me less or equal count of cells')
        return [1] * length + [0] * (self._total_cell_count() - length)

    def _generate_image(self,
                        sequence: List[int] = None):
        sequence = deepcopy(sequence)
        random.shuffle(sequence)
        sequence = np.array(sequence).reshape(self.grid_size)
        h, w = self.grid_size[0] * self.inner_cell_size[0], self.grid_size[1] * self.inner_cell_size[1]
        image = np.zeros((h, w), dtype=np.uint8)
        for i, j in product(range(self.grid_size[0]), range(self.grid_size[1])):
            if sequence[i, j]:
                y, x = i * self.inner_cell_size[0], j * self.inner_cell_size[1]
                y, x = x + self.inner_cell_size[0] // 2, y + self.inner_cell_size[1] // 2
                r = np.random.randint(1, self.inner_cell_size[0] // 2)
                c = np.random.randint(50, 255)
                cv2.circle(image, center=(x, y), radius=r, thickness=-1, color=c)
        return image

    def generate(self,
                 length: Optional[int] = None):
        sequence = self._random_sequence(length)
        image = self._generate_image(sequence)
        if self.image_size:
            image = cv2.resize(image, self.image_size)
        return image, np.array(sequence)


if __name__ == '__main__':
    g = Generator(grid_size=(5, 5), image_size=(1000, 1000))

    while True:
        image, sequence = g.generate()
        print(sequence)

        cv2.imshow('image', image)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break