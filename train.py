import math
import random
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
from PIL import Image

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

params = {'batch_size': 100,
          'shuffle': True,
          'num_workers': 6
          }

imgs = []
labels = []
# Code contribution goes to
# https://github.com/SullyChen/Autopilot-TensorFlow
def read_dataset():
    with open('driving_dataset/data.txt') as data:
        for line in data:
            img = f"driving_dataset/{line.split()[0]}"
            angle = math.radians(float(line.split()[1]))
            imgs.append(img)
            labels.append(angle)

    with open('07012018/data.txt') as data:
        for line in data:
            img = f"07012018/data/{line.split()[0]}"
            angle = math.radians(float(line.split()[1].split(",")[0]))
            imgs.append(img)
            labels.append(angle)
    
    return imgs, labels

class SelfDrivingDataset(Dataset):
    def __init__(self):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.imgs, self.labels = read_dataset()
        self.transform = \
            transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.imgs[index]
        label = self.labels[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        if self.transform:
            self.transform(img_as_img)
        return img_as_img, label

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    dataset = SelfDrivingDataset()
    dataset_size = len(dataset)
    train_set, val_set = \
    torch.utils.data.random_split(dataset, [int(0.8* dataset_size), int(0.2*dataset_size)])
    