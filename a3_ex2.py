import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from a3_ex1 import RandomImagePixelationDataset
import torch


def stack_with_padding(batch_as_list: list):
    img, _, _, _ = zip(*batch_as_list)
    max_img_width = max(_.shape[2] for _ in img)
    max_img_height = max(_.shape[1] for _ in img)
    pixelated_images_1, known_arrays_1, target_arrays, image_files = [], [], [], []

    for pixelated_image, known_array, target_array, image_file in batch_as_list:
        padded_pixelated_image = torch.zeros((max_img_height, max_img_width))
        padded_pixelated_image[:pixelated_image.shape[1], :pixelated_image.shape[2]] = torch.from_numpy(pixelated_image)
        pixelated_images_1.append(padded_pixelated_image)

        padded_known_array = torch.ones((max_img_height, max_img_width))
        padded_known_array[:known_array.shape[1], :known_array.shape[2]] = torch.from_numpy(known_array)
        known_arrays_1.append(padded_known_array)

        target_arrays.append(torch.tensor(target_array))

        image_files.append(image_file)
    pixelated_image = torch.stack(pixelated_images_1).unsqueeze(1)
    known_arrays = torch.stack(known_arrays_1).unsqueeze(1)
    return pixelated_image, known_arrays, target_arrays, image_files


if __name__ == "__main__":
    ds = RandomImagePixelationDataset(
        r"your_path",
        width_range=(50, 300),
        height_range=(50, 300),
        size_range=(10, 50)
    )
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=stack_with_padding)
    for (stacked_pixelated_images, stacked_known_arrays, target_arrays, image_files) in dl:
        fig, axes = plt.subplots(nrows=min(dl.batch_size, len(stacked_pixelated_images)), ncols=3)
        for i in range(min(dl.batch_size, len(stacked_pixelated_images))):
            axes[i, 0].imshow(stacked_pixelated_images[i][0], cmap="gray", vmin=0, vmax=255)
            axes[i, 1].imshow(stacked_known_arrays[i][0], cmap="gray", vmin=0, vmax=1)
            axes[i, 2].imshow(target_arrays[i][0], cmap="gray", vmin=0, vmax=255)
        fig.tight_layout()
        plt.show()
