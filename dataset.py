import pandas as pd
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image


class ImageNetDataset(Dataset):
    def __init__(self, root_directory, dataset="data/train_dataset.csv", transform=None):
        self.root_directory = root_directory
        self.image_directory = pd.read_csv(dataset)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_directory)
    
    def __getitem__(self, idx: int) -> tuple:
        image_path = self.image_directory.iloc[idx, self.image_directory.columns.get_loc("Path")]
        label = self.image_directory.iloc[idx, self.image_directory.columns.get_loc("label")]
        image = Image.open(self.root_directory + image_path).convert('RGB')  

        if self.transform:
            image = self.transform(image)
        if isinstance(image, tuple): # handles Test 5crop case
            image = list(image)
            for i in range(len(image)):
                image[i] = image[i].float() / 255.0 # normalize rgb 
        else: #train case
            image = image/255.0
        return image, label