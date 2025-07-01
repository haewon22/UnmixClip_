import torch
import numpy as np
from PIL import Image

class Cutout(object):
    """Randomly mask out one or more patches from a PIL image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (PIL Image): 이미지
        Returns:
            PIL Image: 일부 영역이 0으로 마스킹된 이미지
        """
        np_img = np.array(img)

        if len(np_img.shape) == 2:
            np_img = np_img[:, :, None]

        h, w, c = np_img.shape

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            np_img[y1:y2, x1:x2, :] = 0

        np_img = np_img.astype(np.uint8)

        if c == 1:
            img = Image.fromarray(np_img[:, :, 0], mode='L')
        else:
            img = Image.fromarray(np_img, mode='RGB')

        return img
