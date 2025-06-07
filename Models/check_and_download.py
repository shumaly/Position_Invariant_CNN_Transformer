import os
import gdown  

# Define file paths and Google Drive file IDs
files = {
    "files/images_data_all.pkl": "1vWs72KClY6kNmHdjFylOuGtmr7pYql5h",
    "logs/best_model_weights.pth": "1VeDwQeGkbeDTgMqgIW3ruEBBkf24u2e5"
}

# Ensure necessary directories exist
os.makedirs("logs", exist_ok=True)

# Function to check and download missing files
def check_and_download():
    for filepath, file_id in files.items():
        if not os.path.exists(filepath):
            print(f"⚠️ {filepath} not found. Downloading from Google Drive...")
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            gdown.download(url, filepath, quiet=False)
            print(f"✅ Downloaded and saved to {filepath}")
        else:
            print(f"✔️ {filepath} already exists. No download needed.")



