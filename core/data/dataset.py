import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .processor import TKRDataProcessor

class TKRDataset(Dataset):
    def __init__(
        self, 
        csv_path: str, 
        processor: TKRDataProcessor,
        transform=None,
        validate_files: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.transform = transform

        if validate_files:
            self.df = self._filter_valid_files(self.df)
    
    def _filter_valid_files(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check image paths in dataFrame, remove problem image path.
        """
        print(f"[Dataset]\t Checking {len(df)} files validity...")
        
        valid_indices = []
        missing_count = 0
        
        for idx in tqdm(df.index, desc="Validating images"):
            img_path = df.loc[idx, "image_path"]
            
            if not os.path.exists(img_path):
                missing_count += 1
                print(f"[Dataset]\t [Warning] Missing file: {img_path}")
                continue
            
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception:
                missing_count += 1
                print(f"[Dataset]\t [Warning] Corrupt file: {img_path}")
                continue
                
            valid_indices.append(idx)
            
        if missing_count > 0:
            print(f"[Dataset]\t ⚠️ Removed {missing_count} invalid/missing images. Remaining: {len(valid_indices)}")
        print(f"\n")
            
        return df.loc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row: pd.Series = self.df.iloc[idx]
        
        # 1. Load Image
        img_path = row["image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"[Dataset]\t Error loading image: {img_path}, {e}")
            image = torch.zeros((3, 224, 224)) 

        # 2. Process Tabular Features
        features = self.processor.process_features(row)
        
        # 3. Process Label
        label_str = row[self.processor.target_col]
        label = self.processor.process_label(label_str)
        
        # 4. Return Metadata (Sample ID)
        metadata = {
            "image_path": img_path,
            "patient_id": str(row.get("unique_patient_id", ""))
        }
        
        return image, features, label, metadata