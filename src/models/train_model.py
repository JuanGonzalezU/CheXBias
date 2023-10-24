import torch
import os
import argparse
import sys
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

# Add general functions
sys.path.append('/home/juandres/aml/CheXBias/src/')
from general_functions import *


# Get and check arguments --------------------------------------------------------------------------------

# Get arguments
parser = argparse.ArgumentParser()

# Add option of sending classes
def list_of_strings(arg):
    return arg.split(',')

# Default values
default_classes = ['Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion','Pleural Other','Fracture']

# Add agruments

# Clases to be clasified
parser.add_argument('--classes',
                    type=list_of_strings,
                    default=default_classes,  
                    help='Classes to be used for training')

# Batches
parser.add_argument('--batch_size',
                    type=int,
                    default=8,
                    help = 'Training batch size')


# Get all arguments
args = parser.parse_args()

# Sanitiy check if arguments

# Classes
for ind_class in args.classes:
    if ind_class not in default_classes:
        print(bcolors.FAIL + f"Error in argument. Class '{ind_class}' in --classes, doesn't exists."+ bcolors.ENDC )
        sys.exit()
        
# Set device
device = 'cuda' if torch.cuda.is_available else 'cpu'

# Load and transform data --------------------------------------------------------------------------------

# Data loader

# csv file with labels
csv_file = "/home/juandres/aml/CheXBias/data/raw/CheXpert-v1.0/train_VisualCheXbert.csv"

# dir to raw data
root_dir = "/home/juandres/aml/CheXBias/data/interim/"

# Define transformation
preprocess = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



# Create custon dataset
custom_dataset = CustomImageDataset(csv_file=csv_file, root_dir=root_dir, classes=args.classes, transform=preprocess)

# Create data loader
data_loader = DataLoader(custom_dataset,batch_size=args.batch_size)

# Define model --------------------------------------------------------------------------------

model = models.densenet121(pretrained=True).to(device)

# Replace input layer to have gray scale
#model.features[0] = nn.Conv2d(1, 256, kernel_size=7, stride=2, padding=3, bias=False).to(device)

# Replace output layer with number of classes
#model.classifier = nn.Linear(model.classifier.in_features, out_features=len(args.classes)).to(device)

# Train model --------------------------------------------------------------------------------

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Number of training epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, inputs, labels in enumerate(data_loader):
        breakpoint()
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 1 == 0:  # Print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')