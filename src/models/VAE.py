
import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import sys

try: 

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
                        default=0.0001,
                        help='Models learning rate')
        
    # Sex proportion
    parser.add_argument('--sex_proportion',
                        type=list_of_ints,
                        default=None,
                        help='Proportion of Male/Female in training data (E.g. 40/60)')

    # Define who is working on the code
    parser.add_argument('--my_name',
                        type = str,
                        default = 'Juan',
                        help = 'Name of person that runs this script (Juan OR Sebastian)')

    # Get all arguments
    args = parser.parse_args()

    args.num_output_channels = 1

    # Default model location
    if args.my_name == 'Juan':
        dir_models = '/home/juandres/aml/CheXBias/models/'
        sys.path.append('/home/juandres/aml/CheXBias/src/')
    else:
        dir_models = '/media/disk2/srodriguez47/ProyectoAML/CheXBias/models'
        sys.path.append('/media/disk2/srodriguez47/ProyectoAML/CheXBias/src/')

    # Import general functions
    from general_functions import *

    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the VAE
    vae = AdaptableVAE(input_channels=1, latent_size=128, input_size=224).to(device)
    
    # Define the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.lr)

    # Load data
    data_loader_train, data_loader_val = train_test_dataloaders_sex(args)
    
    # Log interval
    log_interval = 20  # Adjust this based on your preference

    # Training loop
    for epoch in range(args.epochs):

        # Set model in trianing mode
        vae.train()

        # Loop over batches
        for batch_idx, (_,data,y) in enumerate(data_loader_train):
            
            # Move data to device
            data = data.to(device)
            y = y.to(device)

            # Forward pass
            recon_batch, mu, log_var, y_pred = vae(data)
            
            # Compute the loss
            loss = loss_function_VAE(recon_batch, data, y, y_pred, mu, log_var)

            # Zero grad the optimizer
            optimizer.zero_grad()
            
            # Backward pass and optimization
            loss.backward()

            # Step the optimizer
            optimizer.step()

            # Print training statistics
            #if batch_idx % log_interval == 1:            
            print(f'Train Epoch: {epoch+1}/{args.epochs} [{batch_idx * len(data)}/{len(data_loader_train.dataset)}] Loss: {loss.item()/len(data)}')
            if batch_idx % log_interval == 0:
                torch.cuda.empty_cache()

    # Optionally, you can save the trained model
    if args.my_name == 'Juan':
        torch.save(vae.state_dict(), '/home/juandres/aml/CheXBias/models/VAE/'+args.save)
        pass
    else:
        torch.save(vae.state_dict(), '/media/disk2/srodriguez47/ProyectoAML/CheXBias/models/test.pth')

except KeyboardInterrupt:
    #breakpoint()
    print("Cleaning up...")
    torch.cuda.empty_cache()