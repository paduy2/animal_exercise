import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
import pickle
import numpy as np
import cv2
from PIL import Image


class CIFARDataset(Dataset):
    def __init__(self, root="data", is_train=True):
        if is_train:
            data_files = [os.path.join(root, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            data_files = [os.path.join(root, "test_batch")]
        self.images = []
        self.labels = []
        for data_file in data_files:
            with open(data_file, 'rb') as fo:
                data = pickle.load(fo, encoding='bytes')
                self.images.extend(data[b'data'])
                self.labels.extend(data[b'labels'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]
        image = np.reshape(image, (3, 32, 32))
        # image = np.transpose(image, (1, 2, 0))

        return image, label


class AnimalDataset(Dataset):
    def __init__(self, root, is_train, transform=None):
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider",
                           "squirrel"]
        if is_train:
            data_path = os.path.join(root, "train")
        else:
            data_path = os.path.join(root, "test")

        self.images = []
        self.labels = []

        for idx, category in enumerate(self.categories):
            category_path = os.path.join(data_path, category)
            for item in os.listdir(category_path):
                image_path = os.path.join(category_path, item)
                self.images.append(image_path)
                self.labels.append(idx)

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # image = cv2.imread(self.images[idx])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]

        return image, label


if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])
    dataset = AnimalDataset(root="data/animals", is_train=True, transform=transform)
    # image, label = dataset[100]
    # print(image.shape)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    # for images, labels in dataloader:
    #     print(images.shape, labels)
