# Create excel with results
import os
import sys
import argparse
sys.path.append('/home/juandres/aml/CheXBias/src/')
from general_functions import *


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
                            default=60,
                            help = 'Training batch size')

    # Sub sampler
    parser.add_argument('--subsampler',
                        type=float,
                        default=0.001,
                        help = 'Percentage of data to be used')


    # Number of workers
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Number of workers for the dataloader')
        
        
    # Sex proportion
    parser.add_argument('--sex_proportion',
                        type=list_of_ints,
                        default=[50,50],
                        help='Proportion of Male/Female in training data (E.g. 40/60)') 

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

    # Define cuda device to use
    parser.add_argument('--cuda_device',
                        type=str,
                        default='1')                        
                
    # Get all arguments
    args = parser.parse_args()

    # Add number output channels
    args.num_output_channels = 3

    # Path to experiment results
    path_results = '/home/juandres/aml/CheXBias/models/'

    # Set device
    device = 'cuda:'+args.cuda_device if torch.cuda.is_available else 'cpu'

    # List all experiments
    all_folders = os.listdir(path_results)

    # Experiments
    experiments = []
    # Get only experiment folders
    for experiment in all_folders:
        if 'Experiment' in experiment:
            experiments.append(experiment)

    # Path to save prediction as a report
    path_report = '/home/juandres/aml/CheXBias/reports'

    # Iterate over each experiment
    for experiment in ['Experiment_2']:

        # Get experiment path
        exp_path = os.path.join(path_report,experiment)

        # Create one folder for storing results per experiment
        if not os.path.exists(exp_path):
            
            # Directory for experiment
            os.mkdir(exp_path)

            # Directory for experiment        
            [os.mkdir(os.path.join(exp_path,group)) for group in ['sex','age']]
        
        # Get groups per experiment
        groups = os.listdir(os.path.join(path_results,experiment))

        # Iterate over each group
        for group in ['age']:#groups:

            # Path of model's folder
            path_model = os.path.join(path_results,experiment,group)

            # List on all the models in the group
            models = os.listdir(os.path.join(path_model))
            models.sort()

            # Iterate over each model and create predictions
            for name_model in models:

                # Create structure for an empty model            
                model = CustomResNet(num_classes=len(default_classes))

                # Load weights
                model.load_state_dict(torch.load(os.path.join(path_model,name_model,'best_model.pth'),map_location=device))    

                # Load model to device                
                model.to(device)

                # Get all the test data for sex.

                # Set arguments for dataloade (only loda the test data)
                if group == 'sex':
                    _ , data_loader_test = train_test_dataloaders_sex(args)
                else:
                    _ , data_loader_test = train_test_dataloaders_age(args) 

                # Change model to eval mode 
                model.eval()

                # Initalize variables
                total_acc = []
                total_precision = []
                total_recall = []
                total_f1 = []

                # Turn on inference context manager
                with torch.inference_mode():
                    
                    # All metrics
                    all_metrics_mean = []
                    all_metrics_std = []

                    if group == 'sex':                  
                        
                        # Create directory for saving results
                        dir_to_model_report = os.path.join(path_report,experiment,group,name_model)
                        if not os.path.isdir(dir_to_model_report):
                            os.mkdir(dir_to_model_report)                        
                            for indv_sex in ['male','female']:
                                    dir_to_model_by_sex_report = os.path.join(dir_to_model_report,indv_sex) 
                                    os.mkdir(dir_to_model_by_sex_report)
                                    # Create empty csv with headers
                                    metric_names = ['accuracy','precision','recall','f1']
                                    for metric_name in metric_names:
                                        # Name for csv file
                                        path_csv = os.path.join(dir_to_model_by_sex_report,metric_name+'.csv')
                                        with open(path_csv,'w',newline = '') as file:
                                            # Create writer
                                            writer = csv.writer(file)
                                            # Write header
                                            writer.writerow(args.classes)                        
                        
                        # Iterate in dataloader
                        for batch,(file_names,X,y) in tqdm(enumerate(data_loader_test)):  

                                # Move data to device
                                X, y = X.to(device), y.to(device)              
                                
                                # Binarized sex (True: Female, False: Male)
                                bool_sex = [file_name.split('_')[4] == 'Female' for file_name in file_names]                
                                
                                # Create predictions
                                y_pred = model(X)

                                # Separate predictions based on bool_sex
                                
                                # Prediction
                                y_pred_female = torch.stack([y_pred[i] for i in range(len(y_pred)) if bool_sex[i]],dim=0)
                                y_pred_male = torch.stack([y_pred[i] for i in range(len(y_pred)) if not bool_sex[i]],dim=0)

                                # Ground truth
                                y_female = torch.stack([y[i] for i in range(len(y)) if bool_sex[i]],dim=0)
                                y_male = torch.stack([y[i] for i in range(len(y)) if not bool_sex[i]],dim=0)

                                # Calculate metrics per each class                
                                accuracy_male, precision_male, recall_male, f1_male = calculate_metrics(y_male, y_pred_male)
                                accuracy_female, precision_female, recall_female, f1_female = calculate_metrics(y_female, y_pred_female)

                                all_metrics_male = [accuracy_male,precision_male,recall_male,f1_male]
                                all_metrics_female = [accuracy_female,precision_female,recall_female,f1_female]

                                # Save results on each batch of data
                                for indv_sex in ['male','female']: 
                                    # Directory by sex
                                    dir_to_model_by_sex_report = os.path.join(dir_to_model_report,indv_sex)                                                             
                                    metric_names = ['accuracy','precision','recall','f1']
                                    for i,metric_name in enumerate(metric_names):
                                        path_csv = os.path.join(dir_to_model_by_sex_report,metric_name+'.csv')
                                        with open(path_csv,'a',newline = '') as file:
                                            writer = csv.writer(file)
                                            if indv_sex == 'male':
                                                writer.writerow(all_metrics_male[i].tolist())
                                            else:
                                                writer.writerow(all_metrics_female[i].tolist())                            

                    elif group == 'age':                    
                        # Create directory for saving results                        
                        dir_to_model_report = os.path.join(path_report,experiment,group,name_model)
                        if not os.path.isdir(dir_to_model_report):
                            os.mkdir(dir_to_model_report)                        
                            # Create empty csv with headers
                            metric_names = ['predictions','ground_truth']
                            for metric_name in metric_names:
                                # Name for csv file
                                path_csv = os.path.join(dir_to_model_report,metric_name+'.csv')
                                with open(path_csv,'w',newline = '') as file:
                                    # Create writer
                                    writer = csv.writer(file)
                                    # Write header
                                    writer.writerow(['name'] + args.classes)

                        # Iterate in dataloader
                        for batch,(file_names,X,y) in tqdm(enumerate(data_loader_test)):

                            # Send data to device
                            X, y = X.to(device), y.to(device)                            

                            #  Encode files by age
                            ages = [int(file_name.split('_')[5]) for file_name in file_names]  
                            
                            # Create predictions
                            y_pred = model(X)

                            # Move to CPU
                            y_pred = torch.round(y_pred).detach().cpu().numpy()
                            y = torch.round(y).detach().cpu().numpy()

                            # Save predictions                            
                            with open(os.path.join(dir_to_model_report,'predictions.csv'),'a',newline = '') as file:
                                writer = csv.writer(file)                                
                                for name_count, row in enumerate(y_pred):                                    
                                    # Data to write                                                    
                                    writer.writerow([file_names[name_count]] + row.astype(int).tolist())
                            
                            with open(os.path.join(dir_to_model_report,'ground_truth.csv'),'a',newline = '') as file:
                                writer = csv.writer(file)                                
                                for name_count, row in enumerate(y):                                    
                                    # Data to write                                                    
                                    writer.writerow([file_names[name_count]] + row.astype(int).tolist())

except KeyboardInterrupt:
    print("Cleaning up...")
    torch.cuda.empty_cache()
                                        

                                    





                                

                                
                                

                    

                


            