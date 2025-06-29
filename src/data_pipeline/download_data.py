# src/data_pipeline/download_data.py

import kagglehub

def download_dataset():
    path = kagglehub.dataset_download("rabieelkharoua/parkinsons-disease-dataset-analysis")
    print("Dataset downloaded at:", path)
    return path

if __name__ == "__main__":
    download_dataset()