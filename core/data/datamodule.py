import os
import json
import torch
import numpy as np
import random
import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from ..mix import get_mix_fn

from .dataset import TKRDataset
from .processor import TKRDataProcessor
from .transforms import build_transforms

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class TKRDataModule:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.data_cfg = cfg.data
        
        self.runtime_dir = os.path.join(self.cfg.output_dir, "runtime_data")
        os.makedirs(self.runtime_dir, exist_ok=True)
        
        self.split_dir = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.processor = None

    def setup(self, processor_path: str | None = None):
        """
        Fit processor & build datasets.
        """
        print("[DataModule]\t Setting up DataModule...")
        
        # 1. Read CSV
        train_csv_path, val_csv_path, test_csv_path = self._read_csv()
        
        # 2. Initialize & fit processor
        self._fit_or_load_tabular_processor(train_csv_path, processor_path)

        # 3. Build transforms
        train_transform = build_transforms(self.data_cfg, is_train=True)
        eval_transform = build_transforms(self.data_cfg, is_train=False)
        
        # 4. Build datasets
        self.build_datasets(train_csv_path, val_csv_path, test_csv_path, train_transform, eval_transform)

        # 5. Show preprocessed images
        if self.cfg.get("debug", {}).get("see_preprocessed_img", False):
            self.visualize_train_batch(get_mix_fn(self.cfg))

        if self.train_ds is not None and self.val_ds is not None and self.test_ds is not None:
            print(f"[DataModule]\t Data Loaded: Train={len(self.train_ds)}, Val={len(self.val_ds)}, Test={len(self.test_ds)}")
        else:
            raise ValueError(f"[DataModule]\t [Error]\t train dataset: {self.train_ds} | validation dataset: {self.val_ds} | testing dataset: {self.test_ds}")

    def _read_csv(self):
        '''
        Read & create **runtime** csv files like train.csv, val.csv and test.csv.<br>
        ("runtime" means replacing image_path with paths in image id map)

        Returns:
            train_csv_path, val_csv_path, test_csv_path
        '''
        id_map_mode = self.data_cfg.get("image_id_map_mode", "original")
        map_path = self.data_cfg.image_id_map_file.get(id_map_mode)

        with open(map_path, "r", encoding="utf-8") as f:
            image_map = json.load(f)
            print(f"[Data Module]\t Using image id map from {map_path}")
        
        version = self.data_cfg.get("split_version")
        if version is None:
            raise ValueError(f"[DataModule]\t [Error]\t Setting up data module without assigning datasets split version")
        
        self.split_dir = os.path.join("data/TKR/splits", version)

        train_csv_path = self._create_runtime_csv("train.csv", image_map)
        val_csv_path   = self._create_runtime_csv("val.csv", image_map)
        test_csv_path  = self._create_runtime_csv("test.csv", image_map)

        return train_csv_path, val_csv_path, test_csv_path

    def _fit_or_load_tabular_processor(self, train_csv_path: str, processor_path: str | None = None):
        '''
        fit processor and save it if processor_path wasn't provided.
        '''
        if processor_path is None:

            print(f"[DataModule]\t Initializing and fitting new processor...")

            df_train = pd.read_csv(train_csv_path)
            self.processor = TKRDataProcessor(self.data_cfg)
            self.processor.fit(df_train)

            # Save processor for later (inference) use
            os.makedirs(self.cfg.output_dir, exist_ok=True)
            self.processor.save(os.path.join(self.cfg.output_dir, "preprocessor.pkl"))

        else:
            print(f"[DataModule]\t Loading fitted processor from {processor_path}")
            self.processor = TKRDataProcessor.load(processor_path)

    def build_datasets(self, train_csv_path, val_csv_path, test_csv_path, train_transform, eval_transform):
        '''
        Builds datasets from csv paths.<br>
        Would check if self.processor is TKRDataProcessor
        '''
        if not isinstance(self.processor, TKRDataProcessor):
            raise ValueError(f"[DataModule]\t [Error]\t Build datasets without fitting or loading TKRDataProcessor")

        self.train_ds = TKRDataset(train_csv_path, self.processor, transform=train_transform)
        self.val_ds = TKRDataset(val_csv_path, self.processor, transform=eval_transform)
        self.test_ds = TKRDataset(test_csv_path, self.processor, transform=eval_transform)

    def _create_runtime_csv(self, filename: str, image_map: dict) -> str:
        """
        Update "image_path" in original splitted csv,<br>
        save to runtime_dir.

        Returns:
            path_to_updated_csv(str):
        """
        if self.split_dir is None:
            print(f"[Data Module]\t [Warning]\t self.split_dir is None")
            source_path = os.path.join(filename)
        else:
            source_path = os.path.join(self.split_dir, filename)
        
        target_path = os.path.join(self.runtime_dir, filename)

        print(f"[DataModule]\t source {filename}: {source_path}")
        
        df = pd.read_csv(source_path)
        
        target_col = self.data_cfg.labels.target_col
        filter_classes = self.data_cfg.get("filter_classes", None)

        if filter_classes:
            filter_list = list(filter_classes)
            original_len = len(df)
            df = df[df[target_col].isin(filter_list)]
            print(f"[DataModule]\t [{filename}] Applied filter_classes {filter_list}. Kept {len(df)/original_len} rows.")

        df["image_path"] = df["unique_img_id"].map(image_map)

        mapping = self.data_cfg.get("label_mapping", None)

        if mapping:
            mapping_dict = dict(mapping)
            df[target_col] = df[target_col].map(mapping_dict).fillna(df[target_col])
            print(f"[DataModule]\t [{filename}] Applied label mapping to column '{target_col}'.")
        
        missing_mask = df["image_path"].isna()
        missing_count = missing_mask.sum()
        
        if missing_count > 0:
            print(f"[DataModule]\t [{filename}] Found {missing_count} samples without cropped images (YOLO failed).")
            
            original_len = len(df)
            df = df.dropna(subset=["image_path"]).reset_index(drop=True)
            new_len = len(df)
            
            print(f"[DataModule]\t -> Data reduced from {original_len} to {new_len}.")
        
        df.to_csv(target_path, index=False, encoding="utf-8-sig")
        return target_path

    def get_cls_freq_list(self):
        """
        Returns:
            list: [count_of_class_0, count_of_class_1, ...]
        """
        if self.train_ds is None:
            raise RuntimeError("DataModule not setup yet. Call setup() first.")

        target_col = self.data_cfg.labels.target_col
        label_counts = self.train_ds.df[target_col].value_counts()
        num_classes = self.data_cfg.labels.num_classes
        freq_list = [0] * num_classes

        for label_str, count in label_counts.items():
            label_idx = self.processor.process_label(label_str).item()

            if label_idx >= 0 and label_idx < num_classes:
                freq_list[label_idx] = count
            else:
                print(f"[DataModule]\t [Warning] Found label index {label_idx} out of range (0-{num_classes-1})")
                
        print(f"[DataModule]\t Computed Class Frequencies: {freq_list}")
        return freq_list

    def _get_oversampler(self, dataset):
        df = dataset.df
        target_col = self.processor.target_col
        print("[DataModule]\t -> Calculating sampler weights...")

        all_labels = []
        for label_str in df[target_col]:
            label_idx = self.processor.process_label(label_str).item()
            all_labels.append(label_idx)
        all_labels = torch.tensor(all_labels, dtype=torch.long)

        class_counts = torch.bincount(all_labels)
        class_weights = 1. / class_counts.float()
        sample_weights = class_weights[all_labels] # cool, use c++ parallel computing!

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True,
        )

        return sampler

    def get_train_loader(self, batch_size=None):
        bs = batch_size if batch_size is not None else self.data_cfg.batch_size
        use_oversampling = self.data_cfg.get("use_oversampling", False)
        sampler = None
        shuffle = True

        if use_oversampling:
            print("[DataModule]\t Oversampling Enabled: Using WeightedRandomSampler.")
            sampler = self._get_oversampler(self.train_ds)
            shuffle = False    

        print(f"[DataModule]\t training set: {len(self.train_ds)}")

        g = torch.Generator()
        g.manual_seed(self.cfg.seed)

        return DataLoader(
            self.train_ds,
            batch_size=bs,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    def get_val_loader(self, batch_size=None):

        print(f"[DataModule]\t val set: {len(self.val_ds)}")

        g = torch.Generator()
        g.manual_seed(self.cfg.seed)

        bs = batch_size if batch_size is not None else self.data_cfg.batch_size
        return DataLoader(
            self.val_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    def get_test_loader(self, batch_size=None):
        bs = batch_size if batch_size is not None else self.data_cfg.batch_size

        g = torch.Generator()
        g.manual_seed(self.cfg.seed)

        return DataLoader(
            self.test_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    def visualize_train_batch(self, mixup_fn=None):
        """
        Call this right after dm.setup() to visually inspect augmentations.
        """
        print("=" * 50, " Showing prepeocessed images ", "=" * 50)
        print(f"[DataModule]\t Mixup function: {mixup_fn}")

        train_loader = self.get_train_loader()
        images, metadata, labels, _ = next(iter(train_loader))
        
        if mixup_fn is not None:
            images, metadata, labels = mixup_fn(images, metadata, labels)

        # De-normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images_denorm = images * std + mean
        images_denorm = torch.clamp(images_denorm, 0, 1)

        # Plot
        grid = vutils.make_grid(images_denorm[:16], nrow=4, padding=2)
        plt.figure(figsize=(12, 12))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.title("Preprocessed Training Batch (First 16)")
        plt.axis("off")
        plt.show()