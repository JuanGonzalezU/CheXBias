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
                        default=8,
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

    # Data loader

    # csv file with labels
    csv_file = "/home/juandres/aml/CheXBias/data/raw/CheXpert-v1.0/train_VisualCheXbert.csv"
    csv_file_val = '/home/juandres/aml/CheXBias/data/raw/CheXpert-v1.0/valid.csv'

    # dir to raw data
    root_dir_train = "/home/juandres/aml/CheXBias/data/interim/train/"
    root_dir_val = "/home/juandres/aml/CheXBias/data/interim/val/"

    # Pre-processing transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create index for random sub samples

    # List all files
    all_files_train = os.listdir(root_dir_train)

    # Get a random subset of files based on the proportion

    # Shuffle all files
    random.shuffle(all_files_train)

    # Get a random number of files to be used 
    num_files_to_use = int(len(all_files_train) * args.subsampler)

    # Redefine all files
    all_files_train = all_files_train[:num_files_to_use]

    # All files val
    all_files_val = os.listdir(root_dir_val)

    # Create custon dataset
    custom_dataset_train = CustomImageDataset(csv_file=csv_file, root_dir=root_dir_train, classes=args.classes, transform=preprocess, all_files=all_files_train)
    custom_dataset_val = CustomImageDataset(csv_file=csv_file_val, root_dir=root_dir_val, classes=args.classes, transform=preprocess, all_files=all_files_val)

    # Create data loader
    data_loader_train = DataLoader(custom_dataset_train,batch_size=args.batch_size, num_workers=args.num_workers)
    data_loader_val = DataLoader(custom_dataset_train,batch_size=args.batch_size, num_workers=args.num_workers)

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
        print(f"Epoch: {epoch}\n---------")
        
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
