import os
import pickle
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings
import argparse

warnings.filterwarnings("ignore")

# External module to check and download necessary files.
from Models import check_and_download
check_and_download.check_and_download()

# Import custom modules from the inference_components package.
from inference_components import data_preprocessing, dataset, model

def main(args):
    ''' 
    Main function to perform inference using a pretrained model.
    
    Args:
        args: Parsed command-line arguments.
    '''
    # Fixed parameters for pretrained model inference.
    sliding_window = 20
    img_size = (int(72 * 3), int(33 * 3))
    
    # Load the images dictionary from the specified files directory.
    images_dict = data_preprocessing.load_images_dict(args.files_dir)
    
    # Build the training cube to compute normalization factors.
    train_params, train_label, train_pos_not_normal, df_train_all = data_preprocessing.build_train_cube(args.files_dir, sliding_window)
    
    # Build the test cube using training normalization factors.
    test_params, test_label, max_verpix, max_horpix = data_preprocessing.build_test_cube(args.files_dir, sliding_window, train_pos_not_normal)
    
    # Create the test dataset and DataLoader.
    add_imgs = 0
    test_dataset = dataset.CustomDataset(test_params, test_label, img_size, sliding_window,
                                         add_imgs, max_verpix, max_horpix, aug=0, images_dict=images_dict)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    batch_size_value = args.batch_size
    
    # Configuration for the model (defaults replicate the current pretrained setup).
    config = {
        "Data_shape": [batch_size_value, 512, sliding_window],
        "emb_size": 64,
        "num_heads": 4,
        "dim_ff": 128,
        "Fix_pos_encode": "ldAPE",
        "Rel_pos_encode": "eRPE",
        "dropout": 0,
        "num_labels": 1
    }
    max_label = np.max(train_label)
    min_label = np.min(train_label)
    
    # Initialize the custom model.
    custom_model = model.CustomModel(img_size, sliding_window, max_label, min_label, config)
    
    # Load the saved model weights.
    model_weights_path = os.path.join(args.logs_dir, "best_model_weights.pth")
    model_state = torch.load(model_weights_path, map_location=torch.device('cpu'))
    custom_model.load_state_dict(model_state)
    custom_model.eval()
    
    # Perform inference and compute RMSE.
    criterion = nn.MSELoss()
    valid_epoch_loss = 0.0
    outputs_list = []
    labels_list = []
    
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data  # Unpack test data.
        image_input = inputs['image_input'].float()
        number_input = inputs['number_input'].float()
        labels = labels.float()
    
        outputs = custom_model(image_input, number_input)
        outputs_list += list(outputs.cpu().detach().numpy() * 10000)
        labels_list += list(labels.cpu().detach().numpy() * 10000)
        loss = criterion(outputs, labels)
        valid_epoch_loss += loss.item()
    
    # Plot predictions vs. ground truth.
    plt.figure(figsize=(10, 6))
    plt.plot(outputs_list, label='Estimation', color="red", linestyle='--')
    plt.plot(labels_list, label='Ground Truth', color="black", linestyle='-')
    plt.legend()
    plt.title('Estimation vs. Ground Truth')
    plt.xlabel('Frames')
    plt.ylabel('Drop width (um)')
    plt.show()
    
    # Calculate and display RMSE.
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(outputs_list, labels_list))
    print("RMSE:", rmse)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference using the pretrained model.")
    parser.add_argument("--files_dir", type=str, default="files", 
                        help="Directory containing the input files (Excel and pickle files).")
    parser.add_argument("--logs_dir", type=str, default="logs", 
                        help="Directory containing model weights and logs.")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for the DataLoader during inference.")
    args = parser.parse_args()
    main(args)

