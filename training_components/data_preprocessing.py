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
        files_dir (str): Directory containing the image pickle file.
        
    Returns:
        dict: Dictionary with all images.
    '''
    pkl_path = os.path.join(files_dir, 'images_data_all.pkl')
    with open(pkl_path, 'rb') as f:
        images_dict = pickle.load(f)
    return images_dict

def load_dataframe(files_dir):
    '''
    Load and prepare the dataframe from the Excel file.
    
    Args:
        files_dir (str): Directory containing the Excel file.
        
    Returns:
        pd.DataFrame: Dataframe with a new column "img_names" computed.
    '''
    df = pd.read_excel(os.path.join(files_dir, "dataset_full.xlsx"))
    df = df[["Video ID", "sequence", "x_center_full", "status", "front_width"]]
    df["img_names"] = df["Video ID"].astype(str) + ".0_" + df["sequence"].astype(str) + ".tif"
    return df

def compute_image_dimensions(images_dict):
    '''
    Compute the maximum vertical and horizontal dimensions from the images.
    
    Args:
        images_dict (dict): Dictionary of images.
        
    Returns:
        tuple: (max_verpix, max_horpix)
    '''
    dim0_list, dim1_list = [], []
    for img in images_dict.values():
        shape_img = img.shape
        dim0_list.append(shape_img[0])
        dim1_list.append(shape_img[1])
    max_verpix = max(dim0_list)
    max_horpix = max(dim1_list)
    return max_verpix, max_horpix

def get_train_test_dfs(df):
    '''
    Split the dataframe into training and testing subsets based on the "status" column.
    
    Args:
        df (pd.DataFrame): Full dataframe.
        
    Returns:
        tuple: (df_train_all, df_test_all)
    '''
    df_train_all = df[df["status"] == "train"]
    df_test_all = df[df["status"] == "test"]
    return df_train_all, df_test_all

def build_train_cube(df_train_all, sliding_window):
    '''
    Build sliding window cubes for training data.
    
    Args:
        df_train_all (pd.DataFrame): Dataframe for training samples.
        sliding_window (int): Number of consecutive frames to use.
        
    Returns:
        tuple: (train_params, train_label, train_pos_not_normal)
            - train_params (np.array): Normalized training parameters.
            - train_label (np.array): Training labels.
            - train_pos_not_normal (np.array): Original positional differences.
    '''
    df_train = df_train_all[["img_names", "front_width", "x_center_full"]]
    df_train_slide = np.zeros([1, sliding_window, 3])
    video_ids = list(set(df_train_all["Video ID"]))
    for vid in video_ids:
        df_train_video = df_train[df_train_all["Video ID"] == vid]
        x_center_temp = df_train_video.x_center_full
        diffs = [x_center_temp.iloc[i+1] - x_center_temp.iloc[i] for i in range(len(x_center_temp) - 1)]
        # Replace x_center_full with differences (first value is set to 0).
        df_train_video["x_center_full"] = np.insert(diffs, 0, 0, axis=0)
        df_video_slide = slide_window(sliding_window, df_train_video)
        df_train_slide = np.concatenate((df_train_slide, df_video_slide), axis=0)
    train_add = df_train_slide[1:, :, 0]
    train_pos_not_normal = df_train_slide[1:, :, 2]
    train_pos = df_train_slide[1:, :, 2]
    # Normalize positional differences using training data.
    train_pos = (train_pos - np.min(train_pos_not_normal.reshape(-1, 1))) * 2 / (
        np.max(train_pos_not_normal.reshape(-1, 1)) - np.min(train_pos_not_normal.reshape(-1, 1))
    ) - 1
    train_params = np.concatenate((np.expand_dims(train_add, axis=2),
                                   np.expand_dims(train_pos, axis=2)), axis=2)
    train_label = df_train_slide[1:, :, 1][:, int(sliding_window/2)].reshape(-1, 1)
    return train_params, train_label, train_pos_not_normal

def build_test_cube(df_test_all, sliding_window, train_pos_not_normal):
    '''
    Build sliding window cubes for test data.
    
    Args:
        df_test_all (pd.DataFrame): Dataframe for test samples.
        sliding_window (int): Number of consecutive frames to use.
        train_pos_not_normal (np.array): Original training positional differences for normalization.
        
    Returns:
        tuple: (test_params, test_label)
            - test_params (np.array): Normalized test parameters.
            - test_label (np.array): Test labels.
    '''
    df_test = df_test_all[["img_names", "front_width", "x_center_full"]]
    df_test_slide = np.zeros([1, sliding_window, 3])
    video_ids = list(set(df_test_all["Video ID"]))
    for vid in video_ids:
        df_test_video = df_test[df_test_all["Video ID"] == vid]
        x_center_temp = df_test_video.x_center_full
        diffs = [x_center_temp.iloc[i+1] - x_center_temp.iloc[i] for i in range(len(x_center_temp) - 1)]
        df_test_video["x_center_full"] = np.insert(diffs, 0, 0, axis=0)
        df_video_slide = slide_window(sliding_window, df_test_video)
        df_test_slide = np.concatenate((df_test_slide, df_video_slide), axis=0)
    test_add = df_test_slide[1:, :, 0]
    df_test_slide[1:, :, 2][df_test_slide[1:, :, 2] < -0.2] = 0
    test_pos = df_test_slide[1:, :, 2]
    test_pos = (test_pos - np.min(train_pos_not_normal.reshape(-1, 1))) * 2 / (
        np.max(train_pos_not_normal.reshape(-1, 1)) - np.min(train_pos_not_normal.reshape(-1, 1))
    ) - 1
    test_params = np.concatenate((np.expand_dims(test_add, axis=2),
                                  np.expand_dims(test_pos, axis=2)), axis=2)
    test_label = df_test_slide[1:, :, 1][:, int(sliding_window/2)].reshape(-1, 1)
    return test_params, test_label

