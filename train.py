import math
import random
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
from PIL import Image
from torch.optim.lr_scheduler import MultiStepLR
from model import Dave2Model 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


batch_size = 100
num_workers = 4
epochs = 30

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
    model = 
    dataset = SelfDrivingDataset()
    dataset_size = len(dataset)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8* dataset_size), int(0.2*dataset_size)])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    lr = 1e-4
    weight_decay = 1e-5
    lr_scheduler = MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)
    optimizer = optim.Adam(model.parameters(),
                       lr=lr,
                       weight_decay=weight_decay)
    criterion = nn.MSELoss()
    train_loss = 0.
    for epoch in range(epochs):
        for i, imgs, labels in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
        
            with torch.no_grad():
                for val_imgs, val_labels in val_loader:
                    val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)
