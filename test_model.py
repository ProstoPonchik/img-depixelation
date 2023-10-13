import torch
import torch.nn as nn
from a3_ex1 import RandomImagePixelationDataset
from a3_ex2 import stack_with_padding
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = RandomImagePixelationDataset(
    r"fix images",
    width_range=(4, 32),
    height_range=(4, 32),
    size_range=(4, 16)
)


def test_model(model_path, weight_path):
    model = torch.load(model_path)

    model.load_state_dict(torch.load(weight_path))
    model.eval()

    model = model.to(device)
    criterion = nn.MSELoss()

    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=stack_with_padding)

    with torch.no_grad():
        total_val_loss = 0
        for stacked_pixelated_images, stacked_known_arrays, target_arrays, image_files in loader:

            stacked_pixelated_images = stacked_pixelated_images / 255
            stacked_pixelated_images = stacked_pixelated_images.to(device)
            stacked_known_arrays = stacked_known_arrays.to(device)
            concatenated = torch.cat((stacked_pixelated_images, stacked_known_arrays), dim=1)
            concatenated = concatenated.to(device)

            output = model(concatenated.to(torch.float32))

            local_loss = 0
            for i, target_array in enumerate(target_arrays):
                output_mask = output[i][stacked_known_arrays[i] < 1]
                target_array = torch.flatten(target_array / 255).to(device)
                local_loss += criterion(output_mask, target_array)

            batch_loss = local_loss / len(target_arrays)
            total_val_loss += batch_loss
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            pixelated_image = stacked_pixelated_images.detach().cpu().numpy()[0]
            pixelated_image = pixelated_image.squeeze()
            axs[0].imshow(pixelated_image, cmap='gray')
            axs[0].set_title("Пиксель")

            target_img = target_arrays[0].detach().cpu().numpy()
            target_img = target_img.squeeze()
            axs[1].imshow(target_img, cmap='gray')
            axs[1].set_title("Нужный")

            output_img = output.detach().cpu().numpy()[0]
            output_img = output_img.squeeze()
            axs[2].imshow(output_img, cmap='gray')
            axs[2].set_title("Выход")

            plt.show()
        print("Валидольные потери:", total_val_loss / len(loader))


if __name__ == "__main__":
    test_model("v4/model.pth", "v4/best_model_weights_validation.pth")
