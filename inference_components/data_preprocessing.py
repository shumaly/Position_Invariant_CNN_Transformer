import os
import pickle
import cv2
import numpy as np
import pandas as pd
from Models.utils import slide_window  

def load_images_dict(files_dir):
    ''' 
    Load the pre-saved images dictionary from a pickle file.
    
    Args:
        files_dir (str): Directory where the files are located.
        
    Returns:
        dict: Dictionary containing image data.
    '''
    pkl_path = os.path.join(files_dir, 'images_data_example.pkl')
    with open(pkl_path, 'rb') as f:
        images_dict = pickle.load(f)
    return images_dict

def build_train_cube(files_dir, sliding_window=20):
    '''
    Build the training cube to compute normalization factors using the training dataset.
    
    Args:
        files_dir (str): Directory containing the Excel and pickle files.
        sliding_window (int): Window size for the sliding window.
        
    Returns:
        tuple: (train_params, train_label, train_pos_not_normal, df_train_all)
            - train_params (np.array): Normalized training parameters.
            - train_label (np.array): Training labels.
            - train_pos_not_normal (np.array): Original positional differences before normalization.
            - df_train_all (pd.DataFrame): Training data frame for reference.
    '''
    # Load the dataset from Excel.
    df = pd.read_excel(os.path.join(files_dir, "dataset_full.xlsx"))
    df = df[["Video ID", "sequence", "x_center_full", "status", "front_width"]]
    
    # Generate image file names based on Video ID and sequence.
    list_file_names = []
    for i in range(len(df)):
        file_name = str(df["Video ID"].iloc[i]) + ".0_" + str(df["sequence"].iloc[i]) + ".tif"
        list_file_names.append(file_name)
    df_img_names = pd.DataFrame(list_file_names, columns=["img_names"])
    df = df.join(df_img_names)
    
    # Filter for training data.
    df_train_all = df[df["status"] == "train"]
    df_train = df_train_all[["img_names", "front_width", "x_center_full"]]
    
    # Build the training cube using a sliding window.
    df_train_slide = np.zeros([1, sliding_window, 3])
    Video_ID_train_set = list(set(df_train_all["Video ID"]))
    for vid in Video_ID_train_set:
        df_train_video = df_train[df_train_all["Video ID"] == vid]
        x_center_temp = df_train_video.x_center_full
        # Compute frame-to-frame differences.
        x_center_diff = [x_center_temp.iloc[i+1] - x_center_temp.iloc[i] for i in range(len(x_center_temp)-1)]
        # Modify the dataframe to include frame differences.
        df_train_video["x_center_full"] = np.insert(x_center_diff, 0, 0, axis=0)
        df_train_video_slide = slide_window(sliding_window, df_train_video)
        df_train_slide = np.concatenate((df_train_slide, df_train_video_slide), axis=0)
    
    train_add = df_train_slide[1:, :, 0]
    # Set positional differences below -0.2 to 0.
    df_train_slide[1:, :, 2][df_train_slide[1:, :, 2] < -0.2] = 0
    train_pos_not_normal = df_train_slide[1:, :, 2]
    train_pos = df_train_slide[1:, :, 2]
    # Normalize the positional difference using training data.
    train_pos = (train_pos - min(train_pos_not_normal.reshape(-1, 1))) * 2 / (
        max(train_pos_not_normal.reshape(-1, 1)) - min(train_pos_not_normal.reshape(-1, 1))
    ) - 1
    train_params = np.concatenate(
        (np.expand_dims(train_add, axis=2), np.expand_dims(train_pos, axis=2)),
        axis=2
    )
    train_label = df_train_slide[1:, :, 1][:, int(sliding_window/2)].reshape(-1, 1)
    
    return train_params, train_label, train_pos_not_normal, df_train_all

def build_test_cube(files_dir, sliding_window, train_pos_not_normal):
    '''
    Build the test cube using file addresses from side_data and front_data Excel files.
    
    Args:
        files_dir (str): Directory containing the test Excel files.
        sliding_window (int): Window size for the sliding window.
        train_pos_not_normal (np.array): Original training positional differences for normalization.
        
    Returns:
        tuple: (test_params, test_label, max_verpix, max_horpix)
            - test_params (np.array): Normalized test parameters.
            - test_label (np.array): Test labels.
            - max_verpix (int): Maximum vertical pixels for image framing.
            - max_horpix (int): Maximum horizontal pixels for image framing.
    '''
    df_side = pd.read_excel(os.path.join(files_dir, "side_data.xlsx"))
    df_front = pd.read_excel(os.path.join(files_dir, "front_data.xlsx"))
    df_side['width'] = df_front['contact_line_length']
    df = df_side.copy()
    df = df[["number", "x_center", "width"]]
    df = df.rename(columns={"number": "sequence",
                            "x_center": "x_center_full",
                            "width": "front_width"})
    list_file_names = []
    for i in range(len(df)):
        file_name = str(df["sequence"].iloc[i]) + ".tif"
        list_file_names.append(file_name)
    df_img_names = pd.DataFrame(list_file_names, columns=["img_names"])
    df = df.join(df_img_names)
    
    max_verpix = 89 
    max_horpix = 205
    df_test = df[["img_names", "front_width", "x_center_full"]]
    
    df_test_slide = np.zeros([1, sliding_window, 3])
    df_test_video = df_test.copy()
    x_center_temp = df_test_video.x_center_full
    x_center_diff = [x_center_temp.iloc[i+1] - x_center_temp.iloc[i] for i in range(len(x_center_temp)-1)]
    df_test_video["x_center_full"] = np.insert(x_center_diff, 0, 0, axis=0)
    df_test_video_slide = slide_window(sliding_window, df_test_video)
    df_test_slide = np.concatenate((df_test_slide, df_test_video_slide), axis=0)
    test_add = df_test_slide[1:, :, 0]
    df_test_slide[1:, :, 2][df_test_slide[1:, :, 2] < -0.2] = 0
    test_pos = df_test_slide[1:, :, 2]
    
    # Normalize test positions using training normalization factors.
    test_pos = (test_pos - min(train_pos_not_normal.reshape(-1, 1))) * 2 / (
        max(train_pos_not_normal.reshape(-1, 1)) - min(train_pos_not_normal.reshape(-1, 1))
    ) - 1
    test_params = np.concatenate(
        (np.expand_dims(test_add, axis=2), np.expand_dims(test_pos, axis=2)),
        axis=2
    )
    test_label = df_test_slide[1:, :, 1][:, int(sliding_window/2)].reshape(-1, 1)
    
    return test_params, test_label, max_verpix, max_horpix

