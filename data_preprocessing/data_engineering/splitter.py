import os
import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold

from .format_manager import TKRFormatManager
from .constant import SPLIT_RANDOM_ST, VAL_SET_SIZE, TEST_SET_SIZE, STRATIFY_COL

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, required=True, help="path the directory of processed data")
    parser.add_argument("--img_map_filename", type=str, required=True, help="filename of image id map")
    parser.add_argument("--metadata_filename", type=str, required=True, help="csv filename of metadata")
    parser.add_argument("--split_version", type=str, required=True, help="version name of current split")
    return parser.parse_args()

class TKRDatasetSplitter:
    def __init__(self, split_ver, output_dir, metadata_filename, image_map_filename, random_st, val_size, test_size, stratify_col):
        self.format_manager = TKRFormatManager()

        self.split_ver              = split_ver
        self.output_dir             = output_dir
        self.master_metadata_file   = os.path.join(self.output_dir, metadata_filename)
        self.image_map_file         = os.path.join(self.output_dir, image_map_filename)
        self.random_st              = random_st
        self.val_size               = val_size
        self.test_size              = test_size
        self.stratify_col           = stratify_col

    def run(self):
        print("\n=== Starting Dataset Split (Strategy) ===")
        print(f"Output Directory: {self.output_dir}")
        
        if not os.path.exists(self.master_metadata_file):
            raise FileNotFoundError(f"Master metadata not found at {self.master_metadata_file}. Run builder first.")
            
        df_master = pd.read_csv(self.master_metadata_file, dtype={"sheet_name": str})
        
        with open(self.image_map_file, "r", encoding="utf-8") as f:
            image_map = json.load(f)
            
        print(f"Loaded {len(df_master)} patients/entries from metadata.")
        print(f"Loaded {len(image_map)} valid images from map.")

        train_df, val_df, test_df = self._split_patients(df_master)
        
        print("\n[Processing Splits] Expanding image IDs and mapping paths...")
        train_final = self._process_split_dataframe(train_df, image_map, "train")
        val_final   = self._process_split_dataframe(val_df,   image_map, "val")
        test_final  = self._process_split_dataframe(test_df,  image_map, "test")
        
        self._save_csv(train_final, "train.csv")
        self._save_csv(val_final, "val.csv")
        self._save_csv(test_final, "test.csv")
        
        self._print_statistics(train_final, val_final, test_final)
        
        print("\n=== Dataset Split Complete ===")

    def run_kfold_split(self, n_splits: int = 5, random_state: int = 42):
        print("\n=== Starting K-Fold Split ===")
        print(f"Random State: {random_state}")

        if not os.path.exists(self.master_metadata_file):
            raise FileNotFoundError(f"Master metadata not found at {self.master_metadata_file}. Run builder first.")
            
        df_master = pd.read_csv(self.master_metadata_file, dtype={"sheet_name": str})
        
        with open(self.image_map_file, "r", encoding="utf-8") as f:
            image_map = json.load(f)
            
        print(f"Loaded {len(df_master)} patients/entries from metadata.")
        print(f"Loaded {len(image_map)} valid images from map.")

        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=random_state)
        train_val_idx, test_idx = next(sss.split(df_master, df_master[self.stratify_col]))

        df_test = df_master.iloc[test_idx]
        df_train_val = df_master.iloc[train_val_idx].reset_index(drop=True)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for fold, (train_idx, val_idx) in enumerate(skf.split(df_train_val, df_train_val[self.stratify_col])):
            df_train = df_train_val.iloc[train_idx]
            df_val = df_train_val.iloc[val_idx]

            train_images = self._process_split_dataframe(df_train, image_map, "train")
            val_images   = self._process_split_dataframe(df_val, image_map, "val")
            test_images  = self._process_split_dataframe(df_test, image_map, "test")

            fold_dir = f"data/TKR/splits/{self.split_ver}_seed_{random_state}_fold_{fold}"
            os.makedirs(fold_dir, exist_ok=True)

            self._save_csv(train_images, "train.csv", fold_dir)
            self._save_csv(val_images, "val.csv", fold_dir)
            self._save_csv(test_images, "test.csv", fold_dir)

            self._show_label_distribution(pd.concat([train_images, val_images, test_images]), self.stratify_col)
            print(f"Fold {fold} generated. Train: {len(df_train)} patients, {len(train_images)} images, Val: {len(df_val)} patients, {len(val_images)} images.")
        print(f"\nHoldout Test Set: {len(df_test)} patients {len(test_images)} images.")

    def _split_patients(self, df: pd.DataFrame):
        """
        Two phases:
        1. All -> Dev + Test
        2. Dev -> Train + Val
        """
        stratify_labels = df[self.stratify_col] if self.stratify_col and self.stratify_col in df.columns else None
        
        dev_df, test_df = train_test_split(
            df, 
            test_size=self.test_size, 
            stratify=stratify_labels,
            random_state=self.random_st
        )
        
        if stratify_labels is not None:
            stratify_labels = dev_df[self.stratify_col]

        # 第二階段：切出 Val Set
        # 注意：VAL_SIZE 是佔總體的比例 (ex: 0.15)
        # 在 Dev set 中，Val 的比例應該調整為: 0.15 / (1 - 0.15)
        relative_val_size = self.val_size / (1 - self.test_size)
        
        train_df, val_df = train_test_split(
            dev_df,
            test_size=relative_val_size, 
            stratify=stratify_labels,
            random_state=self.random_st
        )
        
        return train_df, val_df, test_df

    def _process_split_dataframe(self, df: pd.DataFrame, image_map, split_name):
        df = df.copy()
        
        df["照片編號"] = df["照片編號"].apply(self.format_manager.expand_ids)
        df_exploded = df.explode("照片編號").reset_index(drop=True)
        
        df_exploded["unique_img_id"] = (
            df_exploded["sheet_name"].astype(str) + "_" + df_exploded["照片編號"].astype(str)
        )
        
        df_exploded["image_path"] = df_exploded["unique_img_id"].map(image_map)
        
        missing_mask = df_exploded["image_path"].isna()
        missing_count = missing_mask.sum()
        
        if missing_count > 0:
            print(f"  [Warning] {split_name}: Dropped {missing_count} rows due to missing images in map.")
            df_exploded = df_exploded.dropna(subset=["image_path"])
            
        return df_exploded

    def _save_csv(self, df: pd.DataFrame, filename, output_dir=None):
        path = os.path.join(self.output_dir, filename) if output_dir is None else os.path.join(output_dir, filename)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  Saved {filename} ({len(df)} images)")

    def _print_statistics(self, train, val, test):
        print("\n--- Distribution Report ---")
        if self.stratify_col in train.columns:
            total = len(train) + len(val) + len(test)
            print(f"{'Label':<15} | {'Train':<10} | {'Val':<10} | {'Test':<10}")
            print("-" * 55)
            
            labels = sorted(train[self.stratify_col].unique())
            for label in labels:
                t_c = (train[self.stratify_col] == label).sum()
                v_c = (val[self.stratify_col] == label).sum()
                test_c = (test[self.stratify_col] == label).sum()
                print(f"{label:<15} | {t_c:<10} | {v_c:<10} | {test_c:<10}")
        else:
            print("Stratify column not found, skipping distribution report.")

    def _show_label_distribution(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        if col not in df.columns:
            print(f"[BUILDER]\t [ERROR]\t {col} doesn't exist in dataframe.")
            return pd.DataFrame()
        
        counts = df[col].value_counts()
        percentages = df[col].value_counts(normalize=True) * 100

        dist_df = pd.DataFrame({
            "Count": counts,
            "Percentage (%)": percentages.round(2),
        })

        print(dist_df)
        return dist_df

if __name__ == "__main__":
    args = parse_args()

    splitter = TKRDatasetSplitter(
        split_ver=args.split_version,
        output_dir=args.processed_dir,
        metadata_filename=args.metadata_filename,
        image_map_filename=args.img_map_filename,
        random_st=SPLIT_RANDOM_ST,
        val_size=VAL_SET_SIZE,
        test_size=TEST_SET_SIZE,
        stratify_col=STRATIFY_COL,
    )

    splitter.run_kfold_split(random_state=splitter.random_st)