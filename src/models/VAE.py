
import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import sys

# Add general functions
#sys.path.append('/home/juandres/aml/CheXBias/src/')
sys.path.append('/media/disk2/srodriguez47/ProyectoAML/CheXBias/src/')
from general_functions import *
from general_functions import loss_function_VAEv2
from general_functions import AdaptableVAE2

# Get arguments
parser = argparse.ArgumentParser()

# Add option of sending classes
def list_of_strings(arg):
    return arg.split(',')
    
# Function for list of strings
def list_of_ints(arg):
    return list(map(int, arg.split('/')))    

# Default values
default_classes = ['Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion','Pleural Other','Fracture']

# Default model location
#dir_models = '/home/juandres/aml/CheXBias/models/'
dir_models = '/media/disk2/srodriguez47/ProyectoAML/CheXBias/models'
# Add agruments

# Clases to be clasified
parser.add_argument('--classes',
                    type=list_of_strings,
                    default=default_classes,  
                    help='Classes to be used for training')

# Batches
parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    help = 'Training batch size')

# Number of epohcs
parser.add_argument('--epochs',
                    type=int,
                    default=2,
                    help='Number of epochs for training')

# Save model
parser.add_argument('--save',
                    type=str,
                    default='unnamed.pth',
                    help='Set the path for training')

# Sub sampler
parser.add_argument('--subsampler',
                    type=float,
                    default=0.1,
                    help = 'Percentage of data to be used')


# Number of workers
parser.add_argument('--num_workers',
                    type=int,
                    default=8,
                    help='Number of workers for the dataloader')
    
# Learning rate
parser.add_argument('--lr',
                    type=float,
                    default=0.001,
                    help='Models learning rate')
    
# Sex proportion
parser.add_argument('--sex_proportion',
                    type=list_of_ints,
                    default=None,
                    help='Proportion of Male/Female in training data (E.g. 40/60)')

# Get all arguments
args = parser.parse_args()

args.num_output_channels = 1

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the VAE
vae = AdaptableVAE2(input_channels=1, latent_size=64, input_size=224).to(device)

# Define the optimizer
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Load data
data_loader_train, data_loader_val = train_test_dataloaders_sex(args)

# Log interval
log_interval = 1  # Adjust this based on your preference

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    vae.train()
    for batch_idx, (_,data,_) in enumerate(data_loader_train):
        
        data = data.to(device)

        optimizer.zero_grad()

        # Forward pass
        recon_batch, mu, log_var = vae(data)
        
        # Compute the loss
        loss = loss_function_VAEv2(recon_batch, data, mu, log_var)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Print training statistics
        #if batch_idx % log_interval == 1:            
        print(f'Train Epoch: {epoch+1}/{num_epochs} [{batch_idx * len(data)}/{len(data_loader_train.dataset)}] Loss: {loss.item()/len(data)}')

# Optionally, you can save the trained model
torch.save(vae.state_dict(), '/media/disk2/srodriguez47/ProyectoAML/CheXBias/models/test.pth')
