import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
import pandas as pd
import os
from PIL import Image

# Class for colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Create class for custom dataset
class CustomImageDataset(Dataset):
    def __init__(self, 
                 csv_file, # Labels stored 
                 root_dir, # Images stored
                 classes, # All the classes to be included
                 transform=None):     

        # Dictionary with labels   
        self.train_dict = pd.read_csv(csv_file).set_index('Path').to_dict(orient='index')

        # Root directory
        self.root_dir = root_dir

        # Transformations
        self.transform = transform

        # List all files
        self.all_files = os.listdir(root_dir)

        # List with all the target features to be included
        self.classes = classes

    def __len__(self):

        # Len of the dataset
        return len(self.all_files)

    def __getitem__(self, idx):   
        
        # Get file name
        file_name = self.all_files[idx]

        # Split name into parts
        file_name_parts = file_name.split('_')

        # Build name as in dictonary
        name_dict = f'CheXpert-v1.0/train/{file_name_parts[0]}/{file_name_parts[1]}/{file_name_parts[2]}_{file_name_parts[3]}.jpg'

        # Get target of the image
        dict_row = self.train_dict.get(name_dict)

        # Target vector
        target = []
        for i,indv_class in enumerate(self.classes):
            target.append(dict_row[indv_class])

        # Define path to image    
        img_path = os.path.join(self.root_dir,file_name)

        # Read image as PIL Image
        img = Image.open(img_path).convert('L')  # 'L' mode for grayscale

        # Apply transformation if it exists
        # HERE IS THE ERRORRRRR    
        if self.transform:
            img = self.transform(img)

        # Get labels
        return img, torch.tensor(target)

# Create custom densenet
