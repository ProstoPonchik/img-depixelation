import pickle
import numpy as np
from a3_ex1 import RandomImagePixelationDataset
from tqdm import tqdm


def create_pixelated_pkl(image_dir, output_file):
    dataset = RandomImagePixelationDataset(
        f"{image_dir}",
        width_range=(4, 32),
        height_range=(4, 32),
        size_range=(4, 16)
    )
    pixelated_images = []
    known_arrays = []
    num_samples = len(dataset)

    for i in tqdm(range(num_samples)):
        pixelated_images.append(np.array(dataset.__getitem__(i)[0]).astype(np.uint8))
        known_arrays.append(np.array(dataset.__getitem__(i)[1]).astype(np.uint8))

    data_dict = {
        'pixelated_images': pixelated_images,
        'known_arrays': known_arrays
    }

    with open(output_file, "wb") as file:
        pickle.dump(data_dict, file)


if __name__ == "__main__":
    image_dir = "fix images"
    output_file = "images_pix.pkl"
    create_pixelated_pkl(image_dir, output_file)
