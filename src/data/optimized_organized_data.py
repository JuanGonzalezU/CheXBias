import os
from os.path import join as join_path
import shutil
import pandas as pd
from tqdm import tqdm
import time

start = time.process_time()

# Define paths
current_path = '/home/juandres/aml/CheXBias/'
data_path = join_path(current_path, 'data')
raw_data_path = join_path(data_path, 'raw', 'CheXpert-v1.0')

# Load feature dataset and create a dictionary for faster lookups
train_df = pd.read_csv(join_path(raw_data_path, 'valid.csv'))
train_dict = train_df.set_index('Path').to_dict(orient='index')

# Initialize a set to keep track of processed files
processed_files = set()

# Iterate over all folders
for dir in tqdm(os.listdir(join_path(raw_data_path, 'valid'))):
    for study in os.listdir(join_path(raw_data_path, 'valid', dir)):
        for image in os.listdir(join_path(raw_data_path, 'valid', dir, study)):            
            img_name, ext = os.path.splitext(image)
            new_name = f'{dir}_{study}_{img_name}_'

            # Look up the data from the dictionary
            df_row = train_dict.get(join_path('CheXpert-v1.0','valid', dir, study, image))
            #breakpoint()

            if df_row:
                features = [df_row[feature] for feature in ['Sex', 'Age', 'Frontal/Lateral', 'AP/PA']]
                new_name += '_'.join(map(str, features)) + ext

                # Create final file path
                new_file_path = join_path(data_path, 'interim','val', new_name)

                # Copy file if it hasn't been processed before
                if new_file_path not in processed_files:
                    shutil.copyfile(join_path(raw_data_path, 'valid', dir, study, image), new_file_path)
                    processed_files.add(new_file_path)

print(time.process_time() - start)
