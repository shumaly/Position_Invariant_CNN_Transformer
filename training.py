import os
import time
import pickle
import random
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import argparse
from Models import check_and_download
from training_components import data_preprocessing, dataset, model

# Ensure required files are downloaded.
check_and_download.check_and_download()

def main(args):
    '''
    Main function to train the model.
    
    Args:
        args: Parsed command-line arguments.
    '''
    # Global configuration.
    ratio = 1
    sliding_window = args.sliding_window
    batch_size = args.batch_size
    aug_intensity = args.aug_intensity
    
    # Load images dictionary and dataframe.
    images_dict = data_preprocessing.load_images_dict(args.files_dir)
    df = data_preprocessing.load_dataframe(args.files_dir)
    max_verpix, max_horpix = data_preprocessing.compute_image_dimensions(images_dict)
    
    # Split dataframe into training and testing subsets.
    df_train_all, df_test_all = data_preprocessing.get_train_test_dfs(df)
    
    # Build sliding window cubes for training data.
    train_params, train_label, train_pos_not_normal = data_preprocessing.build_train_cube(df_train_all, sliding_window)
    
    # Build sliding window cubes for test data.
    test_params, test_label = data_preprocessing.build_test_cube(df_test_all, sliding_window, train_pos_not_normal)
    
    # Prepare training/validation split.
    img_size = (int(72 * ratio), int(33 * ratio))
    params_train, params_valid, labels_train, labels_valid = train_test_split(
        train_params, train_label, test_size=0.2, random_state=42)
    
    # Create training and validation datasets.
    train_dataset = dataset.CustomDataset(params_train, labels_train, img_size, sliding_window,
                                          max_verpix, max_horpix, aug=True, aug_intensity=aug_intensity, images_dict=images_dict)
    valid_dataset = dataset.CustomDataset(params_valid, labels_valid, img_size, sliding_window,
                                          max_verpix, max_horpix, aug=False, aug_intensity=aug_intensity, images_dict=images_dict)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    
    # Ensure GPU is available.
    if not torch.cuda.is_available():
        raise RuntimeError("This code requires a GPU to run.")
    
    # Get label range.
    max_label = np.max(labels_train)
    min_label = np.min(labels_train)
    
    # Define ConvTran configuration.
    config = {
        "Data_shape": [batch_size, 512, sliding_window],
        "emb_size": args.emb_size,
        "num_heads": args.num_heads,
        "dim_ff": args.dim_ff,
        "Fix_pos_encode": args.fix_pos_encode,
        "Rel_pos_encode": args.rel_pos_encode,
        "dropout": args.dropout,
        "num_labels": 1
    }
    
    # Initialize model.
    custom_model = model.CustomModel(img_size, sliding_window, max_label, min_label, config, vgg_type=args.vgg_type)
    custom_model = custom_model.cuda()
    
    # Loss function and optimizer.
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(custom_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Training hyperparameters.
    num_epochs = args.num_epochs
    early_stopping_patience = args.early_stopping_patience
    tolerance_factor = args.tolerance_factor
    best_valid_loss = np.inf
    epochs_no_improve = 0
    best_model_wts = None
    train_loss_list, val_loss_list, time_list = [], [], []
    os.makedirs(args.logs_dir, exist_ok=True)
    
    # Training loop.
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        start_time = time.time()
        custom_model.train()
        train_epoch_loss = 0.0
        for data in tqdm(train_loader, desc='Training', leave=False):
            inputs, labels = data
            image_input = inputs['image_input'].cuda().float()
            number_input = inputs['number_input'].cuda().float()
            labels = labels.cuda().float()
            outputs = custom_model(image_input, number_input)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()
        custom_model.eval()
        valid_epoch_loss = 0.0
        for data in tqdm(valid_loader, desc='Validation', leave=False):
            inputs, labels = data
            image_input = inputs['image_input'].cuda().float()
            number_input = inputs['number_input'].cuda().float()
            labels = labels.cuda().float()
            outputs = custom_model(image_input, number_input)
            loss = criterion(outputs, labels)
            valid_epoch_loss += loss.item()
        avg_valid_loss = valid_epoch_loss / len(valid_loader)
        val_loss_list.append(avg_valid_loss)
        train_loss_list.append(train_epoch_loss / len(train_loader))
        elapsed_time = time.time() - start_time
        time_list.append(elapsed_time)
        print(f"Validation Loss: {avg_valid_loss:.6f}")
        # Early stopping logic.
        if avg_valid_loss < tolerance_factor * best_valid_loss:
            best_valid_loss = min(best_valid_loss, avg_valid_loss)
            best_model_wts = custom_model.state_dict()
            epochs_no_improve = 0
            save_path = os.path.join(args.logs_dir, "best_model_weights.pth")
            torch.save(custom_model.state_dict(), save_path)
            print(f"Saving best model at epoch {epoch+1} with validation loss: {avg_valid_loss:.6f}")
        else:
            epochs_no_improve += 1
        print("Epochs no improvement:", epochs_no_improve)
        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    # Save training metrics.
    pd.DataFrame({'train_loss': train_loss_list}).to_csv(os.path.join(args.logs_dir, 'train_loss.csv'), index=False)
    pd.DataFrame({'valid_loss': val_loss_list}).to_csv(os.path.join(args.logs_dir, 'valid_loss.csv'), index=False)
    pd.DataFrame({'execution_time': time_list}).to_csv(os.path.join(args.logs_dir, 'execution_time.csv'), index=False)
    
    # Prepare test dataset and loader.
    test_dataset = dataset.CustomDataset(test_params, test_label, img_size, sliding_window,
                                         max_verpix, max_horpix, aug=False, aug_intensity=aug_intensity, images_dict=images_dict)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    
    # Evaluation on test set.
    def eval_test(model, test_loader):
        model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for data, label in test_loader:
                images = data['image_input'].cuda().float()
                number_inputs = data['number_input'].cuda().float()
                labels = label.cuda().float()
                outputs = model(images, number_inputs)
                predictions.extend(outputs.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        predictions = np.array(predictions)
        targets = np.array(targets)
        rmse = np.sqrt(mean_squared_error(targets * 10000, predictions * 10000))
        return '{:.4f}'.format(rmse)
    
    # Load best model weights and evaluate.
    best_model_path = os.path.join(args.logs_dir, "best_model_weights.pth")
    custom_model.load_state_dict(torch.load(best_model_path))
    test_rmse = eval_test(custom_model, test_loader)
    pd.DataFrame({'test_rmse': [test_rmse]}).to_csv(os.path.join(args.logs_dir, 'test_rmse.csv'), index=False)
    
    # Save final checkpoint.
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': best_model_wts,
        'optimizer': optimizer.state_dict(),
        'loss': train_loss_list,
        'val_loss': val_loss_list
    }
    torch.save(checkpoint, os.path.join(args.logs_dir, 'best_model_checkpoint.pth'))
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model with adjustable parameters.")
    parser.add_argument("--files_dir", type=str, default="files", 
                        help="Directory containing input files (Excel and pickle files).")
    parser.add_argument("--logs_dir", type=str, default="logs", 
                        help="Directory to save model weights and logs.")
    parser.add_argument("--sliding_window", type=int, default=20, 
                        help="Sliding window size for cube construction.")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for training.")
    parser.add_argument("--vgg_type", type=str, choices=["modified", "standard"], default="modified",
                        help="Type of VGG8 to use: 'modified' (GELU + Blur) or 'standard' (ReLU, no blur).")
    parser.add_argument("--emb_size", type=int, default=64, 
                        help="Embedding size for ConvTran.")
    parser.add_argument("--num_heads", type=int, default=4, 
                        help="Number of heads for ConvTran.")
    parser.add_argument("--dim_ff", type=int, default=128, 
                        help="Feed-forward dimension for ConvTran.")
    parser.add_argument("--fix_pos_encode", type=str, default="ldAPE",
                        help="Fixed positional encoding type for ConvTran.")
    parser.add_argument("--rel_pos_encode", type=str, default="eRPE",
                        help="Relative positional encoding type for ConvTran.")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout rate for ConvTran.")
    parser.add_argument("--num_epochs", type=int, default=300, 
                        help="Number of training epochs.")
    parser.add_argument("--early_stopping_patience", type=int, default=15, 
                        help="Patience for early stopping.")
    parser.add_argument("--tolerance_factor", type=float, default=1.1, 
                        help="Tolerance factor for early stopping.")
    parser.add_argument("--aug_intensity", type=float, default=1.0,
                        help="Augmentation intensity factor.")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers for DataLoader.")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate for optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for optimizer.")
    args = parser.parse_args()
    main(args)

