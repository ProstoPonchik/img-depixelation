from typing import Optional
import glob
import matplotlib.pyplot as plt
from PIL import Image
from a2_ex1 import to_grayscale
from a2_ex2 import prepare_image
import numpy as np
import os
from tqdm import tqdm
from memory_profiler import profile


class RandomImagePixelationDataset:
    # @profile
    def __init__(
            self,
            image_dir,
            width_range: tuple[int, int],
            height_range: tuple[int, int],
            size_range: tuple[int, int],
            dtype: Optional[type] = None):

        self.image_paths = sorted(
            [
                os.path.abspath(path)
                for path in glob.glob(f"{image_dir}/**/*.jpg", recursive=True)
            ]
        )
        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range
        self.dtype = dtype

        if min(width_range) < 2 or min(height_range) < 2 or min(size_range) < 2:
            raise ValueError("minimum value is smaller than 2")
        if width_range[0] > width_range[1] or height_range[0] > height_range[1] or size_range[0] > size_range[1]:
            raise ValueError("minimum value is greater than the maximum value")

    # @profile
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = np.array(Image.open(image_path), dtype=self.dtype)
        img_gray = to_grayscale(img)

        rng = np.random.default_rng(seed=index)
        width = rng.integers(self.width_range[0], self.width_range[1], endpoint=True)
        height = rng.integers(self.height_range[0], self.height_range[1], endpoint=True)
        width = min(width, img_gray.shape[2] - 1)
        height = min(height, img_gray.shape[1] - 1)

        x = rng.integers(0, img_gray.shape[2] - width, endpoint=True)
        y = rng.integers(0, img_gray.shape[1] - height, endpoint=True)
        size = rng.integers(self.size_range[0], self.size_range[1])
        pixelated_image, known_array, target_array = prepare_image(img_gray, x, y, width, height, size)
        return pixelated_image, known_array, target_array, image_path

    # @profile
    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    ds = RandomImagePixelationDataset(r"your_path", width_range=(4, 32),
                                      height_range=(4, 32),
                                      size_range=(4, 16))

    # # Создание объекта изображения PIL из массива numpy.

    # # Сохранение изображения.
    # im.save('pixelated_image.png')
    # im1.save('pixelated_image.png')
    # im2.save('pixelated_image.png')
    # fig, axes = plt.subplots(ncols=3)
    # axes[0].imshow(pixelated_image[0], cmap="gray", vmin=0, vmax=255)
    # axes[0].set_title("pixelated_image")
    # axes[1].imshow(known_array[0], cmap="gray", vmin=0, vmax=1)
    # axes[1].set_title("known_array")
    # axes[2].imshow(target_array[0], cmap="gray", vmin=0, vmax=255)
    # axes[2].set_title("target_array")
    # fig.suptitle(image_file)
    # fig.tight_layout()
    # plt.show()
