import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
from tqdm import tqdm
import os
from glob import glob
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from a3_ex1 import RandomImagePixelationDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
shuffle = False
predictions = []


def produce_predictions(model_path, model_weight_path, test_data_path):
    model = torch.load(model_path)
    model.load_state_dict(torch.load(model_weight_path))
    model = model.to(device)
    model.eval()

    with open(test_data_path, "rb") as f:
        test_data = pickle.load(f)

    pixelated_images = torch.from_numpy(np.array(test_data["pixelated_images"]))
    known_arrays = torch.from_numpy(np.array(test_data["known_arrays"]))

    dataset = TensorDataset(pixelated_images, known_arrays)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    for batch in tqdm(dataloader):
        batch_pixelated_images, batch_known_arrays = batch
        batch_pixelated_images = batch_pixelated_images / 255
        batch_pixelated_images = batch_pixelated_images.to(device)

        batch_known_arrays = batch_known_arrays.to(device)

        concatenated = torch.cat((batch_pixelated_images, batch_known_arrays), dim=1)
        concatenated = concatenated.to(device)

        with torch.no_grad():
            output = model(concatenated.to(torch.float32))

        for i in range(len(batch_known_arrays)):
            output_mask = output[i][batch_known_arrays[i] < 1]
            predictions.append(output_mask.detach().cpu().numpy().ravel())

    return [np.round(np.clip(pred * 255, 0, 255)).astype(np.uint8) for pred in predictions]


def prepare_image_for_training(image_dir, output_dir,  width_range, height_range, size_range):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = sorted(
        [
            os.path.abspath(path)
            for path in glob(f"{image_dir}/**/*.jpg", recursive=True)
        ]
    )
    for image_path in tqdm(image_paths, desc="Цветное в черное"):
        img = Image.open(image_path).convert('L')

        transform = transforms.Compose([
            transforms.Resize((64, 64), InterpolationMode.BILINEAR),
            transforms.CenterCrop((64, 64)),
        ])

        img_transformed = transform(img)
        disassembly_path = image_path.split(f'\\')

        if not os.path.exists(f"{output_dir}/gray/{disassembly_path[-2]}"):
            os.makedirs(f"{output_dir}/gray/{disassembly_path[-2]}")
        img_transformed.save(f"{output_dir}/gray/{disassembly_path[-2]}/{disassembly_path[-1]}")

    grayscale_to_training_set(f"{output_dir}/gray", output_dir, width_range, height_range, size_range)

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def grayscale_to_training_set(input_directory, output_directory, width_range, height_range, size_range):
    ds = RandomImagePixelationDataset(input_directory, width_range=(width_range[0], width_range[1]),
                                      height_range=(height_range[0], height_range[1]),
                                      size_range=(size_range[0], size_range[1]))

    image_types = ["pixelated", "known", "target"]
    for image_type in image_types:
        create_directory_if_not_exists(f"{output_directory}/{image_type}")

    for pixelated_image, known_array, target_array, image_file in tqdm(ds, desc="Grayscale в тренировку"):
        disassembly_path = image_file.split(f'\\')
        image_name = disassembly_path[-1]
        folder_name = disassembly_path[-2]
        for image_type in image_types:
            create_directory_if_not_exists(f"{output_directory}/{image_type}/{folder_name}")
        Image.fromarray(pixelated_image[0]).save(f"{output_directory}/pixelated/{folder_name}/{image_name}")
        Image.fromarray(known_array[0]).save(f"{output_directory}/known/{folder_name}/{image_name}")
        Image.fromarray(target_array[0]).save(f"{output_directory}/target/{folder_name}/{image_name}")


if __name__ == "__main__":
    prepare_image_for_training(r"path_to_img", r"output_path_for_grayscale", (4, 32), (4, 32), (4, 16))
    # grayscale_to_training_set(r"path_to_grayscale", r"output_path_for_grayscale", (4, 32), (4, 32), (4, 16))
    # result = produce_predictions("ResNet/model.pth", "ResNet/best_model_weights.pth", "test_set.pkl")
    # print(result)
