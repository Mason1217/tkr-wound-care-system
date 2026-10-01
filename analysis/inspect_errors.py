import os
import torch
import pandas as pd
import argparse
from PIL import Image
from torchvision.utils import save_image
from omegaconf import OmegaConf

from core.config import ConfigManager
from core.data.transforms import build_transforms
from core.data.processor import TKRDataProcessor
from core.data.dataset import TKRDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Experiment config path")
    parser.add_argument("--analysis_results", type=str, required=True, help="Path result dir")
    return parser.parse_args()

def denormalize(tensor: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    """Reverse ImageNet normalization so the image looks normal to human eyes."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1) # Ensure pixels are between 0 and 1

def export_error_visuals(config_path: str, csv_path: str, output_folder: str):
    print(f"Loading errors from {csv_path}...")
    df_errors = pd.read_csv(csv_path)
    
    if df_errors.empty:
        print("No errors found in this CSV.")
        return

    # 1. 載入 Config 並建立驗證集專用的 Transform (包含 Pad, Resize, Normalize)
    cfg = ConfigManager(config_path).cfg
    val_transform = build_transforms(cfg.data, is_train=False) # 確保這裡是 cfg.data
    
    os.makedirs(output_folder, exist_ok=True)
    norm_mean = cfg.data.image_transforms.normalize.mean
    norm_std = cfg.data.image_transforms.normalize.std

    print(f"Exporting {len(df_errors)} preprocessed images to {output_folder}...")
    
    for idx, row in df_errors.iterrows():
        img_path = row["image_path"]
        
        # 安全檢查：確保路徑存在且不是 NaN
        if pd.isna(img_path) or not os.path.exists(str(img_path)):
            print(f"  [Skip] Invalid or missing image path: {img_path}")
            continue
            
        # 2. 直接讀取圖片並進行 Transform
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = val_transform(image)
        except Exception as e:
            print(f"  [Error] Failed to process {img_path}: {e}")
            continue
        
        # 3. 反正規化 (Denormalize)
        clean_image = denormalize(image_tensor, norm_mean, norm_std)
        
        # 4. 組合檔名並儲存
        true_label = row["true_label"]
        pred_label = row["pred_label"]
        patient_id = row["patient_id"]
        original_filename = os.path.basename(img_path)
        
        save_name = f"True{true_label}_Pred{pred_label}_{patient_id}_{original_filename}"
        save_path = os.path.join(output_folder, save_name)
        
        save_image(clean_image, save_path)
        
    print("Done! You can now review the images.")

if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    result_dir = args.analysis_results
    
    # 1. Export High Confidence Errors
    export_error_visuals(
        config_path=config_path,
        csv_path=os.path.join(result_dir, "threshold_0.4/high_confidence_errors.csv"),
        output_folder=os.path.join(result_dir, "threshold_0.4/error_visuals/high_confidence"),
    )
    
    # 2. Export Unstable Patients
    export_error_visuals(
        config_path=config_path,
        csv_path=os.path.join(result_dir, "threshold_0.4/unstable_patients_details.csv"),
        output_folder=os.path.join(result_dir, "threshold_0.4/error_visuals/unstable_patients")
    )