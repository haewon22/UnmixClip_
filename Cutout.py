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
        # PIL 이미지를 Numpy 배열로 변환 (H, W, C)
        np_img = np.array(img)

        # 혹시 흑백 이미지인 경우 채널 차원이 없으므로 예외처리
        if len(np_img.shape) == 2:
            # (H, W)을 (H, W, 1)로 변환
            np_img = np_img[:, :, None]

        h, w, c = np_img.shape

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            # 해당 영역을 0으로 만들어서 "cutout" 효과
            np_img[y1:y2, x1:x2, :] = 0

        # uint8로 형변환 후 다시 PIL 이미지로 변환
        np_img = np_img.astype(np.uint8)

        # 원본 이미지가 흑백이었다면 'L', 컬러였다면 'RGB'
        if c == 1:
            img = Image.fromarray(np_img[:, :, 0], mode='L')
        else:
            img = Image.fromarray(np_img, mode='RGB')

        return img
