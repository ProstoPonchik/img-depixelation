import pickle
import numpy as np
from a3_ex1 import RandomImagePixelationDataset
from tqdm import tqdm


def create_target_pkl(image_dir, output_file):
    dataset = RandomImagePixelationDataset(
        f"{image_dir}",
        width_range=(4, 32),
        height_range=(4, 32),
        size_range=(4, 16),
    )

    num_samples = len(dataset)
    target_arr = []

    for i in tqdm(range(num_samples)):
        target_array = dataset.__getitem__(i)[2][:64][0]
        target_arr.append(np.concatenate(target_array))

    with open(output_file, "wb") as file:
        pickle.dump(target_arr, file)


if __name__ == "__main__":
    create_target_pkl("fix images", "images.pkl")
