import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from a3_ex1 import RandomImagePixelationDataset
from a3_ex2 import stack_with_padding
from model import ResNet

from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
start = time.time()
dataset = RandomImagePixelationDataset(
    r"fix images",
    width_range=(4, 32),
    height_range=(4, 32),
    size_range=(4, 16)
)
print(time.time() - start)
train_ratio = 0.8  # 80% for training
val_ratio = 0.2  # 20% for validation

num_samples = len(dataset)
train_size = int(train_ratio * num_samples)
val_size = num_samples - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=stack_with_padding)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=stack_with_padding)

criterion = nn.MSELoss()

model = ResNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

early_stopping_patience = 5
patience_counter = 0
validation_loss_best = float('inf')

epoch_losses_list = []
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    print("Епоха: ", epoch + 1)
    epoch_loss = 0
    start_time = time.time()

    train_tqdm = tqdm(train_loader, total=len(train_loader), desc="ТренировОчка")
    for stacked_pixelated_images, stacked_known_arrays, target_arrays, image_files in train_tqdm:

        target_arrays = [torch.flatten(ta / 255).to(device) for ta in target_arrays]
        stacked_pixelated_images = stacked_pixelated_images / 255
        stacked_pixelated_images = stacked_pixelated_images.to(device)

        stacked_known_arrays = stacked_known_arrays.to(device)

        concatenated = torch.cat((stacked_pixelated_images, stacked_known_arrays), dim=1)
        concatenated = concatenated.to(device)

        output = model(concatenated.to(torch.float32))

        local_loss = 0
        for i, target_array in enumerate(target_arrays):
            output_mask = output[i][stacked_known_arrays[i] < 1]
            local_loss += criterion(output_mask, target_array)

        batch_loss = local_loss / len(target_arrays)
        epoch_loss += batch_loss

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        batch_time = time.time() - start_time
        start_time = time.time()

        train_tqdm.set_postfix({'Время бача': f'{batch_time:.3f}', 'Потери на баче': f'{batch_loss:.3f}'})

    epoch_loss = epoch_loss / len(train_loader)

    epoch_losses_list.append(epoch_loss)

    print(f"Епоха {epoch + 1} потери: ", epoch_loss.item())

    torch.save(model.state_dict(), 'v4/model_weights.pth')
    torch.save(model, 'v4/model.pth')

    with open("v4/final_loss.txt", "a") as file:
        list_str = ' '.join(map(str, epoch_losses_list))
        file.write(list_str + "\n")

    epoch_losses_list = []

    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        for stacked_pixelated_images, stacked_known_arrays, target_arrays, image_files in val_loader:
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

        total_val_loss /= len(val_loader)
        print("Потери на валидации: ", total_val_loss)

        if total_val_loss < validation_loss_best:
            validation_loss_best = total_val_loss
            torch.save(model.state_dict(), 'v4/best_model_weights_validation.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Bad model")
                break
