import os
from os.path import join as join_path
import shutil
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time

start = time.process_time()

def process_image(dir, study, image, train_df, features):
    img_name = image.split('.jpg')[0]
    new_name = dir + '_' + study + '_' + img_name + '_'
    df_row = train_df[train_df['Path'].str.contains(join_path(dir, study, image))][features]
    for i, feature in enumerate(features):
        if i == len(features) - 1:
            el = '.jpg'
        else:
            el = '_'
        new_name += str(df_row[feature].item()) + el

    if not os.path.exists(join_path(data_path, 'interim', new_name)):
        shutil.copyfile(join_path(raw_data_train_path, dir, study, image),
                        join_path(data_path, 'interim', new_name))

def process_folder(dir, train_df, features):
    studies = os.listdir(join_path(raw_data_train_path, dir))
    for study in studies:
        images = os.listdir(join_path(raw_data_train_path, dir, study))
        for image in images:
            process_image(dir, study, image, train_df, features)

def process_folder_wrapper(args):
    dir, train_df, features = args
    return process_folder(dir, train_df, features)

if __name__ == "__main__":
    current_path = '/home/juandres/aml/CheXBias/'
    data_path = join_path(current_path, 'data')
    raw_data_path = join_path(data_path, 'raw', 'CheXpert-v1.0')

    raw_data_train_path = join_path(raw_data_path, 'train')
    raw_train_folders = os.listdir(raw_data_train_path)

    features = ['Sex', 'Age', 'Frontal/Lateral', 'AP/PA']
    train_df = pd.read_csv(join_path(raw_data_path, 'train.csv'))

    # Choose the number of cores to utilize
    num_cores = int(input("Enter the number of cores to use: "))

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Use a regular function instead of lambda for mapping
        args = [(dir, train_df, features) for dir in raw_train_folders]
        results = list(tqdm(executor.map(process_folder_wrapper, args), total=len(args)))

    print(time.process_time() - start)