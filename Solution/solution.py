import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image
import os
from datetime import datetime
import random
from typing import Optional
from PIL import Image
from matplotlib import pyplot as plt

from collections import OrderedDict


class CustomDataSet(Dataset):
    def __init__(self, data_set_folder: str, transform: Optional[v2.Compose] = None, cache_size: int = 100):
        self.data_set_folder = data_set_folder
        self.transform = transform
        self.cache_size = cache_size
        self.image_cache = OrderedDict()
        self.image_pairs = self._build_image_pairs()

    def _build_image_pairs(self) -> list[tuple[str, str, int]]:
        pairs = []

        if not os.path.isdir(self.data_set_folder):
            raise FileNotFoundError("Invalid data set folder path!")

        for first_dir in os.listdir(self.data_set_folder):
            images_dir = os.path.join(self.data_set_folder, first_dir, 'images')
            images = sorted([f for f in os.listdir(images_dir) if f.endswith('.tif')])

            for i in range(len(images) - 1):
                for j in range(i + 1, len(images)):
                    date1 = datetime.strptime(" ".join(images[i].split('_')[2:4]), '%Y %m')
                    date2 = datetime.strptime(" ".join(images[j].split('_')[2:4]), '%Y %m')

                    months_diff = (date2.year - date1.year) * 12 + (date2.month - date1.month)

                    start_path = os.path.join(images_dir, images[i])
                    end_path = os.path.join(images_dir, images[j])

                    pairs.append((start_path, end_path, months_diff))

        return pairs

    def _load_image(self, image_path: str) -> Image.Image:
        if image_path in self.image_cache:
            return self.image_cache[image_path]

        with Image.open(image_path) as img:
            image = img.copy()

        if len(self.image_cache) >= self.cache_size:
            self.image_cache.popitem(last=False)
        self.image_cache[image_path] = image

        return image

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, idx):
        start_path, end_path, time_skip = self.image_pairs[idx]

        start_image = self._load_image(start_path)
        end_image = self._load_image(end_path)

        angle = random.randint(-30, 30)
        rotation_transform = v2.RandomRotation((angle, angle))
        start_image = rotation_transform(start_image)
        end_image = rotation_transform(end_image)

        if self.transform:
            start_image = self.transform(start_image)
            end_image = self.transform(end_image)

        return start_image, end_image, time_skip

    def clear_cache(self):
        self.image_cache.clear()


def create_dataloaders(dataset: Dataset, train_procentage, test_procentage):
    total_count = len(dataset)
    train_size = int(total_count * train_procentage)
    test_size = int(total_count * test_procentage)
    val_size = total_count - train_size - test_size

    train, validation, test = torch.utils.data.random_split(dataset, (train_size, val_size, test_size))

    train_data_loader = DataLoader(train, batch_size=10, shuffle=True)
    validation_data_loader = DataLoader(validation, batch_size=10, shuffle=False)
    test_data_loader = DataLoader(test, batch_size=10, shuffle=False)

    return train_data_loader, validation_data_loader, test_data_loader


class Trainer:
    def __init__(self, model: nn.Module, loss_function, optimizer, device, train_loader, validation_loader, test_loader):
        self.model = model
        self.loss_function = loss_function
        self.device = device
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.train_losses = []
        self.validation_losses = []

    def run(self, n):
        for epoch in range(n):
            train_loss = self.train()
            validation_loss = self.val()

            self.train_losses.append(train_loss)
            self.validation_losses.append(validation_loss)

            print(f"Epoch {epoch}: train loss: {train_loss:.2f}, validation loss: {validation_loss:.2f}")

    def train(self):
        self.model.train()
        total_loss = 0

        for start_imgs, end_imgs, time_skips in self.train_loader:
            start_imgs = start_imgs.to(self.device)
            end_imgs = end_imgs.to(self.device)
            time_skips = time_skips.to(self.device)

            predictions = self.model(start_imgs, time_skips)
            loss = self.loss_function(predictions, end_imgs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def val(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for start_imgs, end_imgs, time_skips in self.validation_loader:
                start_imgs = start_imgs.to(self.device)
                end_imgs = end_imgs.to(self.device)
                time_skips = time_skips.to(self.device)

                predictions = self.model(start_imgs, time_skips)
                loss = self.loss_function(predictions, end_imgs)
                total_loss += loss.item()

        return total_loss / len(self.validation_loader)

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.validation_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.show()


class MyNN(nn.Module):
    def __init__(self, max_time_skip=100):
        super().__init__()

        self.in_channels = 3
        self.img_size = 32

        self.time_embedding = nn.Embedding(max_time_skip + 1, 16)

        self.conv1 = nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        flat_size = 16 * self.img_size * self.img_size

        self.time_proj = nn.Linear(16, flat_size)

        self.fc1 = nn.Linear(flat_size, flat_size // 4)
        self.fc2 = nn.Linear(flat_size // 4, 3 * self.img_size * self.img_size)

        self.relu = nn.ReLU()

    def forward(self, x, time_skip):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = x.view(x.size(0), -1)

        time_embed = self.time_embedding(time_skip)
        time_features = self.relu(self.time_proj(time_embed))

        x = x + time_features

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        x = x.view(-1, 3, self.img_size, self.img_size)

        return x


# init (max_time_skip, self.rel = nn.Embeddings(..)
# forward (s, ts)
# torch.Parameter(, 24, 128, req=True)

mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]

transformations = v2.Compose([
    v2.RandomCrop(32, padding=4),
    v2.RandomHorizontalFlip(),
    v2.RandomErasing(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=mean, std=std),
])


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MyNN().to(device)

    custom_data_set = CustomDataSet('../Dataset', transformations)
    train_loader, validate_loader, test_loader = create_dataloaders(custom_data_set, 0.7, 0.15)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = Trainer(
        model=model,
        loss_function=criterion,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        validation_loader=validate_loader,
        test_loader=test_loader
    )

    num_epochs = 100
    trainer.run(num_epochs)

    trainer.plot_losses()


if __name__ == "__main__":
    main()