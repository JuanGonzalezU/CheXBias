import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
import pandas as pd
import os
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import random
from torchvision import transforms
import csv


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
                 all_files, # Percentage of data to be used
                 transform=None):     

        # Dictionary with labels   
        self.train_dict = pd.read_csv(csv_file).set_index('Path').to_dict(orient='index')

        # Root directory
        self.root_dir = root_dir

        # Transformations
        self.transform = transform

        # List all files
        self.all_files = all_files

        ## List with all the target features to be included
        self.classes = classes

        # Define if it is train or val
        self.type = 'valid' if 'valid' in csv_file else 'train'

    def __len__(self):

        # Len of the dataset
        return len(self.all_files)

    def __getitem__(self, idx):   
        
        # Get file name
        file_name = self.all_files[idx]

        # Split name into parts
        file_name_parts = file_name.split('_')

        # Build name as in dictonary
        name_dict = f'CheXpert-v1.0/{self.type}/{file_name_parts[0]}/{file_name_parts[1]}/{file_name_parts[2]}_{file_name_parts[3]}.jpg'

        # Get target of the image
        dict_row = self.train_dict.get(name_dict)

        # Target vector
        target = []
        
        for _,indv_class in enumerate(self.classes):
            target.append(dict_row[indv_class])

        # Define path to image    
        img_path = os.path.join(self.root_dir,file_name)

        # Read image as PIL Image
        img = Image.open(img_path).convert('L')  # 'L' mode for grayscale

        # Apply transformation if it exists          
        if self.transform:
            img = self.transform(img)

        # Get labels
        return name_dict, img, torch.tensor(target)

# Custom DenseNet         
class CustomDenseNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomDenseNet, self).__init__()
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        #self.densenet = models.densenet121(pretrained=True)

        # Replace input layer to have grayscale
        self.densenet.features[0] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace output layer with number of classes
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, out_features=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.densenet(x)
        output = self.sigmoid(features)
        return output   

    
# General metrics function   
def calculate_metrics(target_tensor, predicted_tensor):
  
    # Round the predicted probabilities to obtain binary predictions
    predicted_binary = torch.round(predicted_tensor).detach().cpu().numpy()

    # Convert tensors to numpy arrays
    target_array = target_tensor.cpu().numpy()

    num_classes = target_array.shape[1]
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(num_classes):
        true_positives = ((predicted_binary[:, i] == 1) & (target_array[:, i] == 1)).sum()
        false_positives = ((predicted_binary[:, i] == 1) & (target_array[:, i] == 0)).sum()
        false_negatives = ((predicted_binary[:, i] == 0) & (target_array[:, i] == 1)).sum()
        true_negatives = ((predicted_binary[:, i] == 0) & (target_array[:, i] == 0)).sum()

        accuracy = (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives) if (true_positives + false_positives + false_negatives + true_negatives) != 0 else 0 
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Return the calculated metrics for each class
    return torch.tensor(accuracies, dtype=torch.float), torch.tensor(precisions, dtype=torch.float), torch.tensor(recalls, dtype=torch.float), torch.tensor(f1_scores, dtype=torch.float)


# Function for creating dataloaders
def train_val_dataloaders(args):

    # CSV files for gt (won't be used at all)
    csv_file_train = "/home/juandres/aml/CheXBias/data/raw/CheXpert-v1.0/train_VisualCheXbert.csv"
    csv_file_val = '/home/juandres/aml/CheXBias/data/raw/CheXpert-v1.0/valid.csv'

    # Root directory of files
    root_dir_train = "/home/juandres/aml/CheXBias/data/interim/train/"
    root_dir_val = "/home/juandres/aml/CheXBias/data/interim/val/"

    # Redefine all files
    all_files_train = os.listdir(root_dir_train)

    # Select subset
    
    # Get a random number of files to be used 
    if args.subsampler == 1:
        pass 
    else:
        
        # Number of files to be used
        num_files_to_use = int(len(all_files_train) * args.subsampler)

        # Redefine all files
        all_files_train = all_files_train[:num_files_to_use]

        # All files val
        all_files_val = os.listdir(root_dir_val)

    # All validation files
    all_files_val = os.listdir(root_dir_val)

    # Pre-processing transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load train and val data
    custom_dataset_train = CustomImageDataset(csv_file=csv_file_train, root_dir=root_dir_train, classes=args.classes, transform=preprocess, all_files=all_files_train)
    custom_dataset_val = CustomImageDataset(csv_file=csv_file_val, root_dir=root_dir_val, classes=args.classes, transform=preprocess, all_files=all_files_val)

    # Create data loader
    data_loader_train = DataLoader(custom_dataset_train,batch_size=args.batch_size, num_workers=args.num_workers)
    data_loader_val = DataLoader(custom_dataset_train,batch_size=args.batch_size, num_workers=args.num_workers)

    return data_loader_train, data_loader_val

# Define pre-processing transformations

def pre_processing():

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# Training step function
def train_step(model, data_loader, loss_fn, optimizer, device):

    # Move model to device
    model.to(device)

    # Initialize variablesnv
    train_loss, total_acc, total_pre, total_recall, total_f1 = 0, 0, 0, 0, 0

    # Iterate over dataloader
    for _, (_, X, y) in tqdm(enumerate(data_loader), total=len(data_loader), desc='Training'):
       

        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate Loss
        loss = loss_fn(y_pred,y)
        train_loss += loss

        # 2.1 Calculate metrics
        accuracy, precision, recall, f1 = calculate_metrics(y, y_pred)
        
        total_acc += torch.mean(accuracy).item()
        total_pre += torch.mean(precision).item()
        total_recall += torch.mean(recall).item()
        total_f1 += torch.mean(f1).item()

        # 3. Optimize zero grad
        optimizer.zero_grad()

        # 4. Backward loss
        loss.backward()

        # 5. Optimizer step
        optimizer.step()        


    # Calcualte average metrics
    train_loss /= len(data_loader)
    total_acc /= len(data_loader)
    total_pre /= len(data_loader)
    total_recall /= len(data_loader)
    total_f1 /= len(data_loader)    

    # Print results
    print(f"Train loss: {train_loss:.5f} | Train Acc: {total_acc:.5f}  | Train Precision: {total_pre:.5f} ")

# Testing step function
def test_step(data_loader, model, device, best_metric, dir_model):
    
    # Move model to device
    model.to(device)

    # Change model to eval mode
    model.eval()

    total_acc, total_pre, total_recall, total_f1 = 0, 0, 0, 0

    # Turn on inference context manager
    with torch.inference_mode():

        # Iterate in dataloader
        for batch, (_,X,y) in tqdm(enumerate(data_loader), total=len(data_loader), desc='Validation'):

            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # Forward pass            
            y_pred = model(X)

            # Calcualte metrics
            accuracy, precision, recall, f1 = calculate_metrics(y, y_pred)

            total_acc += torch.mean(accuracy).item()
            total_pre += torch.mean(precision).item()
            total_recall += torch.mean(recall).item()
            total_f1 += torch.mean(f1).item()
                    
        # Calcualte average metrics
        total_acc /= len(data_loader)
        total_pre /= len(data_loader)
        total_recall /= len(data_loader)
        total_f1 /= len(data_loader)

        print(f"Val Acc: {total_acc:.5f}  | Val Precision: {total_pre:.5f} ")

        # Save model
        if (total_acc > best_metric) and (dir_model != 'none'):
            print("Saving model...")
            torch.save(model.state_dict(), dir_model)

        return total_acc


def get_predictions(model, data_loader, classes, device):
    
    predictions_dict = {}
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad(), tqdm(total=len(data_loader), desc="Processing Batches") as pbar:  # tqdm progress bar
         for name_dict, inputs, targets in data_loader:
             inputs, targets = inputs.to(device), targets.to(device)  # Move data to the appropriate device
             
             # Forward pass
             outputs = torch.round(model(inputs))
 
             # Convert outputs to a dictionary with class labels as keys and predictions as values
             predictions = {}
 
             for i, output in enumerate(outputs):
                 for j, indv_pred in enumerate(output):     
                     prediction_value = int(indv_pred.item())  # Get the prediction value from the output tensor
                     predictions[classes[j]] = prediction_value
                     
                 # Add predictions to the dictionary with name_dict as the key
                 predictions_dict[name_dict[i]] = predictions.copy()  # Make a copy to avoid overwriting in the next iteration
                 
             pbar.update(1)  # Update the tqdm progress bar
             
    return predictions_dict

def save_predictions(predictions_dict, output_csv_path, classes):

    with open(output_csv_path, 'w', newline='') as csvfile:

        # Create csv writer
        csvwriter = csv.writer(csvfile)

        # Write the header row with 'Path' in the first column and class labels in subsequent columns
        csvwriter.writerow(['Path'] + classes)

        # Write the predictions to the CSV file
        for key, values in tqdm(predictions_dict.items(), desc="Writing to CSV", ncols=100):
            # Write the dictionary key (name_dict) as the first column
            row = [key] + [values.get(class_label, 0) for class_label in classes]
            csvwriter.writerow(row)