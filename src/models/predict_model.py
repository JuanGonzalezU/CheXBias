# Create report of the model
import sys
import argparse
import os
from torchvision import transforms

sys.path.append('/home/juandres/aml/CheXBias/src/')
from general_functions import *

try: 

    # Prepare workspace --------------------------------------------------------------------------------

    # Default model location

    dir_models = '/home/juandres/aml/CheXBias/models/'
    dir_results = '/home/juandres/aml/CheXBias/data/processed/'

    # Add option of sending classes
    def list_of_strings(arg):
        return arg.split(',')

    # Default values
    default_classes = ['Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Lung Lesion','Edema','Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion','Pleural Other','Fracture']

    # Add agruments
    parser = argparse.ArgumentParser()

    # Clases to be clasified
    parser.add_argument('--model_name',
                        type=str,
                        default='None',
                        help = 'Choose the model to be analyzed (E.g. desenet121/test.pth)')

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

    # Number of workers for the dataloader
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Number of workers for the dataloader')

    parser.add_argument('--subsampler',
                        type=float,
                        default=1,
                        help = 'Percentage of data to be used')

    # Get all arguments
    args = parser.parse_args()

    # Sanity check arguments
    if args.model_name == 'None':
        print(bcolors.FAIL + f"Choose a model (add --model_name argument)."+ bcolors.ENDC )
        sys.exit()

    # Set device
    device = 'cuda:1' if torch.cuda.is_available else 'cpu'

    # Load model --------------------------------------------------------------------------------

    # Load information of trained models

    # THIS PART SHOULD BE COMPLETED


    # Depending on the model, recreate structure
    if 'densenet121' in args.model_name:

        # Recreate structure
        model = CustomDenseNet(num_classes=12).to(device)

        # Load training parameters
        model.load_state_dict(torch.load(os.path.join(dir_models,args.model_name)))        

    else:
        pass # Here we should add other trained models


    # Load data --------------------------------------------------------------------------------

    # Get data loaders
    data_loader_train, data_loader_val = train_val_dataloaders(args)

    # Make predictions for all images
    predictions_dict = get_predictions(model, data_loader_train, args.classes, device)

    # Save results

    # Definie path for saving
    model_directory = os.path.join(dir_results,args.model_name.split('/')[0])

    # Create directory if it doesn't exists
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # Define csv path
    csv_path = os.path.join(model_directory,args.model_name.split('/')[1].split('.pth')[0]+'.csv',)    

    save_predictions(predictions_dict, csv_path, args.classes)

except KeyboardInterrupt:
    print("Cleaning up...")
    torch.cuda.empty_cache()

