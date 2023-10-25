import torch
import os
import argparse
import sys
from torchvision import transforms
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
import atexit

try: 

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

    # Default model location
    dir_models = '/home/juandres/aml/CheXBias/models/'

    # Add agruments

    # Clases to be clasified
    parser.add_argument('--classes',
                        type=list_of_strings,
                        default=default_classes,  
                        help='Classes to be used for training')

    # Batches
    parser.add_argument('--batch_size',
                        type=int,
                        default=90,
                        help = 'Training batch size')

    # Number of epohcs
    parser.add_argument('--epochs',
                        type=int,
                        default=3,
                        help='Number of epochs for training',
                        )


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
                        default=0.01,
                        help='Models learning rate')

    # Get all arguments
    args = parser.parse_args()

    # Sanitiy check if arguments

    # Classes
    for ind_class in args.classes:
        if ind_class not in default_classes:
            print(bcolors.FAIL + f"Error in argument. Class '{ind_class}' in --classes, doesn't exists."+ bcolors.ENDC )
            sys.exit()

    # Directory name
    if '.pth' in args.save:
        pass
    else:
        print(bcolors.FAIL + f"Error in argument. Model saving name '{args.save}' should have .pth extension."+ bcolors.ENDC )
        sys.exit()
      
    # Set device
    device = 'cuda:1' if torch.cuda.is_available else 'cpu'

    # Load and transform data --------------------------------------------------------------------------------

    # Pre-processing transformations
    preprocess = pre_processing()

    # Get data loaders
    data_loader_train, data_loader_val = train_val_dataloaders(args)

    # Define model --------------------------------------------------------------------------------

    model = CustomDenseNet(num_classes=len(args.classes)).to(device)

    # Train model --------------------------------------------------------------------------------

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop

    # Number of epochs
    epochs = args.epochs

    # Save previous best matric
    best_metric = 0

    for epoch in range(epochs):
        
        # Print epochs
        print(f"---------------\nEpoch: {epoch}")
        
        # Training step
        train_step(data_loader=data_loader_train, 
            model=model, 
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device        
        )

        # Testing step
        best_metric = test_step(data_loader=data_loader_val,
            model=model,
            device=device,
            best_metric=best_metric,
            dir_model=os.path.join(dir_models,args.save)
        )

except KeyboardInterrupt:
    print("Cleaning up...")
    torch.cuda.empty_cache()
