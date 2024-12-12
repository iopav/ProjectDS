import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as F
import torch.nn.functional as F_nn

class OralCancerDataset(Dataset):
    def __init__(self, path_to_csv, path_to_image, train=1, transform=None, val_split=0.2, random_seed=42, full_dataset=False):
        self.path_to_csv = path_to_csv
        self.path_to_image = path_to_image
        self.transform = transform
        self.train = train

        self.data = pd.read_csv(self.path_to_csv)
        
        if not full_dataset:
            train_data, val_data = train_test_split(self.data, test_size=val_split, random_state=random_seed, stratify=self.data['Diagnosis'])
            if self.train:
                self.data = train_data
            else:
                self.data = val_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.path_to_image, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        #label = F_nn.one_hot(torch.tensor(self.data.iloc[idx, 1]), num_classes=2)
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        
        return image, label


class RotationTransform:

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return F.rotate(x, angle)