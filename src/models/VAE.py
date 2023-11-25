
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
                        default=400,
                        help = 'Training batch size')

    # Number of epohcs
    parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        help='Number of epochs for training')

    # Save model
    parser.add_argument('--save',
                        type=str,
                        default='unnamed.pth',
                        help='Set the path for training')

    # Sub sampler
    parser.add_argument('--subsampler',
                        type=float,
                        default=1,
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
    # Age range
    parser.add_argument('--age_range',
                        type=int,
                        default=20,
                        help='Range in years of the groups')   

    # Age range for training
    parser.add_argument('--age_group_selection',
                        type=int,
                        default=0,
                        help='Choose what of the groups to use for training')  

    # Define on what re-grouping to train
    parser.add_argument('--grouping',
                        type=str,
                        default='age',
                        help = 'Choose what subgroup to use for training')  

    # Define name of the experiment
    parser.add_argument('--experiment',
                        type=str,
                        default='2',
                        help='Number of the experiment')

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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Instantiate the VAE
    vae = AdaptableVAE(input_channels=1, latent_size=128, input_size=224).to(device)
    
    # Define the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.lr)

    # Load data
    data_loader_train, data_loader_val = train_test_dataloaders_age(args)
    
    # Log interval
    log_interval = 20  # Adjust this based on your preference

    # Training loop

    # Save the best loss
    best_loss = 1e32

    # Path to results
    path_to_results = '/home/juandres/aml/CheXBias/models'

    for epoch in range(args.epochs):

        # Set model in trianing mode
        vae.train()

        # Sum of all loss
        sum_loss = 0
   
        # Loop over batches
        for batch_idx, (_, data, y) in tqdm(enumerate(data_loader_train),total = len(data_loader_train),desc='Training'):

            # Move data to device
            data = data.to(device)
            y = y.to(device)

            # Forward pass
            recon_batch, mu, log_var, y_pred = vae(data)

            # Compute the loss
            loss = loss_function_VAE(recon_batch, data, y, y_pred, mu, log_var)
            sum_loss += loss

            # Zero grad the optimizer
            optimizer.zero_grad()

            # Backward pass and optimization
            loss.backward()

            # Step the optimizer
            optimizer.step()

        # Calculate average loss
        sum_loss /= len(data_loader_train)
        print(f'Train Epoch: {epoch+1}/{args.epochs} [{batch_idx * len(data)}/{len(data_loader_train.dataset)}] Loss: {sum_loss}')

        # Save mode if it is better
        if sum_loss < best_loss:
              
            # Get path for saving results
            path_model = os.path.join(path_to_results,'Experiment_'+args.experiment,args.grouping,args.save.split('.')[0],'best_vae.pth')    
            # Save model
            torch.save(vae.state_dict(),path_model)  
            # Update best loss
            best_loss = sum_loss

except KeyboardInterrupt:
    print("Cleaning up...")
    torch.cuda.empty_cache()