import pandas as pd
import numpy as np
import torch
import joblib
from omegaconf import DictConfig, ListConfig
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

class TKRDataProcessor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.num_cols = cfg.features.numerical
        self.cat_cols = cfg.features.categorical
        self.target_col = cfg.labels.target_col
        self.ordered_label_names = cfg.labels.get("names", None)

        if isinstance(self.ordered_label_names, ListConfig):
            self.ordered_label_names = list(self.ordered_label_names)

        self.label_map = {}

        # Initialize Scalers / Encoders
        self.scaler = StandardScaler()
        self.cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Label Encoder
        self.label_encoder = LabelEncoder()

        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Only called on training set.
        """
        print("[Processor]\t Fit processor on training data...")
        print(f"[Processor]\t Numerical cols: {self.num_cols}")
        print(f"[Processor]\t Categorical cols: {self.cat_cols}")

        # Fit Numerical
        if self.num_cols:
            self.scaler.fit(df[self.num_cols])
        
        # Fit Categorical Features
        if self.cat_cols:
            self.cat_encoder.fit(df[self.cat_cols])
            
        # Fit Labels (Target)
        if self.ordered_label_names:
            print(f"[Processor]\t Using explicit label order from config: {self.ordered_label_names}")
            self.label_map = {name: i for i, name in enumerate(self.ordered_label_names)}
            self.classes_ = np.array(self.ordered_label_names)
            
            unique_labels_in_data = df[self.target_col].unique()
            unknowns = set(unique_labels_in_data) - set(self.ordered_label_names)
            if unknowns:
                print(f"[Processor]\t [Warning] Found labels in data not in config: {unknowns}")
        else:
            print("[Processor]\t No label order defined in config, using LabelEncoder (auto-sorted).")
            self.label_encoder.fit(df[self.target_col])
            self.classes_ = self.label_encoder.classes_
        
        print(f"[Processor]\t Classes found: {self.classes_}")
        self.is_fitted = True

    def process_features(self, row: pd.Series) -> torch.Tensor:
        """
        Handle single feature vector.

        Returns:
            ret(Tensor):
        """
        features = []
        
        # Numerical
        if self.num_cols:
            nums_df = pd.DataFrame([row[self.num_cols].values], columns=self.num_cols)
            nums_scaled = self.scaler.transform(nums_df).flatten()
            features.append(nums_scaled)
            
        # Categorical
        if self.cat_cols:
            cats_df = pd.DataFrame([row[self.cat_cols].values], columns=self.cat_cols)
            cats_encoded = self.cat_encoder.transform(cats_df).flatten()
            features.append(cats_encoded)
            
        if not features:
            return torch.tensor([]) # Image Only case
            
        return torch.tensor(np.concatenate(features), dtype=torch.float32)

    def process_label(self, label_str: str) -> torch.Tensor:
        if self.label_map is not None:
            if label_str not in self.label_map:
                raise ValueError(f"Encountered unknown label: '{label_str}'. Expected: {self.ordered_label_names}")
            label_idx = self.label_map[label_str]
        else:
            label_idx = self.label_encoder.transform([label_str])[0]

        return torch.tensor(label_idx, dtype=torch.long)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)

def df_to_numpy(series: pd.Series):
    return series.to_numpy()