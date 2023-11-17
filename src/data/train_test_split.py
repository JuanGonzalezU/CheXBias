import os
import pandas as pd
import random
import sys
import shutil
from tqdm import tqdm

# path to data
path =  '/home/juandres/aml/CheXBias/data_new/interim/train'
path_raw = '/home/juandres/aml/CheXBias/data_new/raw/CheXpert-v1.0'

# List all files in train
all_files = os.listdir(path)

# Shuffle the files
random.shuffle(all_files)

# Define train test split proportion
train_percentage = 0.8

# Get number of files for train 
num_files_train = len(all_files)*train_percentage
num_files_test = len(all_files)- num_files_train

# Round values
num_files_train, num_files_test = int(num_files_train), int(num_files_test)

# Select train files
train_files = all_files[:num_files_train]
test_files = all_files[num_files_train:]

# Find common files
intersection_files = list(set(train_files).intersection(test_files))

# Find if there is any mistake
if intersection_files != None and len(train_files)+len(test_files) == len(all_files):
    pass
else:
    print('Wrong file selection')
    sys.exit()

# Modify the .csv files

# Load train csv file (all_files)
all_files_labels = pd.read_csv(os.path.join(path_raw,'train_VisualCheXbert.csv'))

# Match file names with label names
train_files_new_name = []
for train_file in train_files:
    # Get each part of the name
    elements = train_file.split('_')
    # Build name
    train_files_new_name.append(f'CheXpert-v1.0/train/{elements[0]}/{elements[1]}/{elements[2]}_{elements[3]}.jpg')

test_files_new_name = []
for test_file in test_files:
    # Get each part of the name
    elements = test_file.split('_')
    # Build name
    test_files_new_name.append(f'CheXpert-v1.0/train/{elements[0]}/{elements[1]}/{elements[2]}_{elements[3]}.jpg')

# Crete df for train and test
train_df = all_files_labels[all_files_labels['Path'].isin(train_files_new_name)]
test_df = all_files_labels[all_files_labels['Path'].isin(test_files_new_name)]

# Copy files from interim/train (all_files) to processed/train

# Path to processed
processed_path = '/home/juandres/aml/CheXBias/data_new/processed/'

# Iterate over train_files_new_name
for train_file in tqdm(train_files):
    # Copy file 
    shutil.copy(os.path.join(path,train_file),os.path.join(processed_path,'train',train_file))    
print('Train files copied!')    

# Iterate over train_files_new_name
for test_file in tqdm(test_files):
    # Copy file
    shutil.copy(os.path.join(path,test_file),os.path.join(processed_path,'test',test_file))
print('Test files copied!') 

# Save the dataframes
path = '/home/juandres/aml/CheXBias/data_new/raw/CheXpert-v1.0'
train_df.to_csv(os.path.join(path,'new_train.csv'),index=False)
test_df.to_csv(os.path.join(path,'new_test.csv'),index=False)

print('Labels in csv files created!')