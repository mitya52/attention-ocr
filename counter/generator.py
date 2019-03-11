import cv2
import numpy as np
from itertools import product
from typing import Tuple


__all__ = ['Generator']


class Generator:
    def __init__(self,
                 grid_size: Tuple[int, int],
                 image_size: Tuple[int, int]):
        self.grid_size = grid_size
        self.image_size = image_size

        self.inner_cell_size = 50, 50

    def _total_cell_count(self) -> int:
        return self.grid_size[0] * self.grid_size[1]

    def _generate_sequence(self,
                           length: int) -> np.array:
        return np.array([0] * length + [1] * (self._total_cell_count() - length), dtype=np.int32)

    def _generate_image(self,
                        sequence: np.array) -> np.array:
        sequence = np.copy(sequence)
        np.random.shuffle(sequence)
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

    def __getitem__(self,
                    length: int) -> Tuple[np.array, np.array]:
        sequence = self._generate_sequence(length)
        image = self._generate_image(sequence)
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
        return image, sequence

    def __next__(self) -> Tuple[np.array, np.array]:
        length = np.random.randint(1, self._total_cell_count())
        return self[length]

    def __iter__(self) -> Tuple[np.array, np.array]:
        while True:
            yield next(self)


class BatchGenerator(Generator):
    def __init__(self,
                 batch_size: int,
                 *args, **kwargs):
        super(BatchGenerator, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.images = np.zeros((self.batch_size, self.image_size[0], self.image_size[1], 1), dtype=np.uint8)
        self.sequences = np.zeros((self.batch_size, self._total_cell_count()), dtype=np.uint8)

    def __next__(self) -> Tuple[np.array, np.array]:
        for idx in range(self.batch_size):
            image, sequence = super(BatchGenerator, self).__next__()
            self.images[idx, :, :, 0] = image
            self.sequences[idx, :] = sequence
        return self.images, self.sequences

    def __iter__(self) -> Tuple[np.array, np.array]:
        while True:
            yield next(self)


if __name__ == '__main__':
    g = BatchGenerator(
        batch_size=1,
        grid_size=(5, 5),
        image_size=(1000, 1000))

    for image, sequence in g:
        print(sequence[0])
        cv2.imshow('image', image[0])
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break