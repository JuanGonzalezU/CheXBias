# This scripts moves all images to data/processed, so the files are not stored under subfolders. Naming structure changes

import os
from os.path import join as join_path
import shutil
import pandas as pd
from tqdm import tqdm
import time

start = time.process_time()

# Define paths
current_path = '/home/juandres/aml/CheXBias/'
data_path = join_path(current_path,'data')
raw_data_path =  join_path(data_path,'raw','CheXpert-v1.0')

# Get all raw data folders
raw_data_train_path = join_path(raw_data_path,'train')
raw_train_folders = os.listdir(raw_data_train_path)

# Features to include 
features = ['Sex','Age','Frontal/Lateral','AP/PA']

# Load feature datasets
train_df = pd.read_csv(join_path(raw_data_path,'train.csv'))

# Iterate over all folders
count_images = 0

for dir in tqdm(raw_train_folders):

    # List all studies
    studies = os.listdir(join_path(raw_data_train_path,dir))

    # Iterate over all studies
    for study in studies:

        # List all images 
        images = os.listdir(join_path(raw_data_train_path,dir,study))

        # Iterate over all images
        for image in images:

            # Get name without extension
            img_name = image.split('.jpg')[0]

            # Raname image
            new_name = dir+'_'+study+'_'+img_name+'_'

            # Here, there will not be filtering conditions, they will be added on the DataLoader. For making it easy, the name will contain relevant information
            # so the change of a bottle neck is reduced. Based on this, some predefined condtions will be added  

            # Create naming convention            
            df_row = train_df[train_df['Path'].str.contains(join_path(dir,study,image))][features]            

            for i,feature in enumerate(features):
                if i == len(features)-1: 
                    el = '.jpg'
                else:
                    el = '_'      
                          
                new_name += str(df_row[feature].item())+el

            # Create final file name

            # Copy file 
            #if not os.path.exists(join_path(data_path,'interim',new_name)):                  
            #    shutil.copyfile(join_path(raw_data_train_path,dir,study,image), join_path(data_path,'interim',new_name))
            
            #count_images += 1
       
print(time.process_time() - start)