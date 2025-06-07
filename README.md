
# CNN-Transformer with Low-Dimensional Absolute Position Encoding for Sliding Drop Width Estimation

This repository provides a deep learning pipeline to estimate the width of sliding drops from side-view video data. Our approach leverages a CNN-Transformer architecture with low-dimensional Absolute Position Encoding (ldAPE) optimized for low-information video datasets, achieving high accuracy with reduced computational overhead.

## Key Contributions

- **VGG8-Inspired Architecture for Low-Information Datasets**  
  We propose a lightweight convolutional backbone optimized for small, low-information density video datasets.

- **ConvTran for Extrinsic Temporal Challenges**  
  We integrate ConvTran, a state-of-the-art time-series analysis model, to address extrinsic temporal challenges. Additionally, we introduce an enhanced Absolute Position Encoding—termed low-dimensional Absolute Position Encoding (ldAPE)—which improves the dot-product mechanism for more accurate drop width estimation.

- **Position-Invariant Frame Processing**  
  Our methodology effectively removes non-essential regions from video frames, enabling the model to focus on the drop shape and reducing computation by 82%.

- **Comprehensive Sliding Drop Dataset**  
  We provide a diverse dataset of sliding drop videos recorded on various surfaces and viscosities, offering a valuable resource for future research.

## Repository Overview

- **Models/**  
  Contains core modules including:
  - `model.py`: Implements components used in the ConvTran architecture, which is integrated into the overall model for effective processing of video data.
  - `augmentation.py`: Provides data augmentation functions (e.g., stain and glow simulation).
  - `utils.py`: Utility functions, such as sliding window processing.

- **inference_components/**  
  Contains the architecture of the CNN-Transformer model used for inference, integrating CNN and Transformer components.

- **training_components/**  
  Contains the architecture of the CNN-Transformer model can beused for training, integrating CNN and Transformer components.

- **Scripts/**  
  - `inference.py`: Command-line script for performing inference using a pretrained model. This script loads the model, prepares the input data, and outputs predictions.
  - `training.py`: Command-line script for training the model from scratch. This script allows you to adjust hyperparameters and save the trained model.

- **requirements.txt**  
  Lists all required Python dependencies.

## Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/...
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you have a GPU-enabled environment (CUDA) if you plan to train the model.*

## Usage

### Training
To train the model, run the following command in your terminal:

```bash
python training.py --files_dir <path_to_files> --logs_dir <path_to_logs> --batch_size <batch_size> --num_epochs <num_epochs>
```

**Key Arguments:**
- `--files_dir`: Directory containing input files (Excel and pickle files).
- `--logs_dir`: Directory to save model weights and logs.
- `--batch_size`: Batch size for training (default: 16).
- `--num_epochs`: Number of training epochs (default: 300).
- Additional arguments can be specified for model configuration (e.g., `--vgg_type`, `--emb_size`, etc.).

### Inference
To perform inference on new data, use the following command:

```bash
python inference.py --files_dir <path_to_files> --logs_dir <path_to_logs> --batch_size <batch_size>
```

**Key Arguments:**
- `--files_dir`: Directory containing the input files (dataset Excel files and pickle files).
- `--logs_dir`: Directory containing model weights and logs.
- `--batch_size`: Batch size for DataLoader during inference (default: 16).

The script will:
1. Load the images and prepare the data for inference.
2. Run inference through the model.
3. Output predictions and visualize results.

### The Dataset
- The dataset is stored in `files/dataset_full.xlsx`, which contains all time-series data preprocessed and extracted using the **4S-SROF method** ([GitHub repository](https://github.com/AK-Berger/4S-SROF)).
- The dataset is linked to `images_data_all.pkl`, which contains images of all side-view droplets.
- The **sequence column + video ID column** in `dataset_full.xlsx` corresponds to the image filenames in `images_data_all.pkl`.
- `images_data_all.pkl` will be downloaded automatically when running the scripts.

## License

This project is under the GPL-3.0 license.

