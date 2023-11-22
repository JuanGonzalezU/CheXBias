# Get joint histogram of the latent features
import pandas as pd
import torch
import sys
import os
from PIL import Image
from torchvision import transforms
from types import SimpleNamespace

sys.path.append('/home/juandres/aml/CheXBias/src/')
from general_functions import *

# Create VAE and load pre-traiened weights

# Set the target GPU
device = torch.device('cuda:1')

# Move the entire process to the specified GPU
torch.cuda.set_device(device)

# Instantiate the VAE
vae = AdaptableVAE(input_channels=1, latent_size=2*64, input_size=224).to(device)

# Model name
model_name = 'test4.pth'

# Path for saving mu values
path_vae_results = '/home/juandres/aml/CheXBias/reports/VAE'

# Load pre-trained weights
vae.load_state_dict(torch.load('/home/juandres/aml/CheXBias/models/VAE/'+model_name, map_location={'cuda:0': 'cuda:1'}))

# Get train data
args = SimpleNamespace(
    subsampler=1,
    sex_proportion=None,
    num_output_channels=1,
    classes=['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture'],
    batch_size=32,
    num_workers=8
)

data_loader_train, _ = train_test_dataloaders_sex(args)
# Iterate over all batches of files for getting the histogram
with torch.inference_mode():   
    
    # Turn model into eval
    vae.eval()

    # Iterate over data with tqdm
    for batch_idx, (names, X, y) in enumerate(tqdm(data_loader_train, desc="Processing Batches")):

        # Move data to device
        X = X.to(device)
        y = y.to(device)

        # Pass data through the encoder
        _, mu, _, _ = vae(X)
        
        # Save these mu values in an excel file
        mu_values = mu.detach().cpu().numpy()

        # Path of the file for saving values
        path_file = os.path.join(path_vae_results, model_name.split('.')[0] + '.csv')

        # Create file if it doesn't exist
        if not os.path.exists(path_file):
            with open(path_file, 'w', newline='') as file:

                # Define headers
                header = ['file_name'] + np.arange(mu_values.shape[1]).astype(str).tolist()
                writer = csv.writer(file)
                
                # Write header
                writer.writerow(header)

        # If it is already created, just append the results
        with open(path_file, 'a', newline='') as file:
            
            # Create writer
            writer = csv.writer(file)

            # Add data
            for i,row in enumerate(mu_values):

                # Data to write                
                writer.writerow([names[i]] + row.tolist())
