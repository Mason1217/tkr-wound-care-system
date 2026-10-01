import os
import glob
import json
import argparse
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from pillow_heif import register_heif_opener

from .format_manager import TKRFormatManager
from .constant import REQUIRED_COLUMNS, RENAME_COLUMNS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_root", type=str, required=True, help="directory contains all raw data")
    parser.add_argument("--raw_metadata_file", type=str, required=True, help="relative path to csv file labeled by nurse")
    parser.add_argument("--raw_img_dir_names", nargs='+', required=True, help="directories (**under data/TKR/**) to be preprocessed")
    parser.add_argument("--output_root", type=str, required=True, help="output directory")
    parser.add_argument("--output_image_dirname", type=str, required=True, help="directory holds processed images")
    parser.add_argument("--output_metadata_filename", type=str, required=True, help="processed metadata filename")
    parser.add_argument("--img_map_filename", type=str, help="filename of image id map")
    return parser.parse_args()

class TKRDatasetBuilder:
    def __init__(self, output_root, img_map_file, raw_image_dirs, raw_metadata_file, master_csv_path, images_out_dir):
        self.format_manager     = TKRFormatManager()

        self.output_root        = output_root
        self.raw_image_dirs     = raw_image_dirs
        self.raw_metadata_file  = raw_metadata_file
        self.required_columns   = REQUIRED_COLUMNS
        self.rename_columns     = RENAME_COLUMNS
        
        # Setup output paths
        self.images_out_dir     = images_out_dir
        self.master_csv_path    = master_csv_path
        self.image_map_path     = img_map_file
        
        os.makedirs(self.images_out_dir, exist_ok=True)

    def run(self):
        print("\n=== Starting Dataset Build (ETL) ===")
        print(f"Output Directory: {self.output_root}")
        
        # 1. Process images (Raw -> Processed JPG)
        image_map = self._process_all_images()
        
        # 2. Process Metadata (Excel -> Cleaned CSV)
        df_master = self._process_metadata()
        
        # 3. Save results
        self._save_results(df_master, image_map)
        print("\n=== Dataset Build Complete ===")

    def _process_all_images(self) -> dict:
        """
        Iterate all raw image directory, convert .heic/.png to .jpg

        Return:
            image_map(dict): {unique_id: absolute_path}
        """
        print(f"\n[1/3] Processing Images from {len(self.raw_image_dirs)} directories...")
        register_heif_opener()
        image_id_map = {}
        
        tasks = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            for input_dir in self.raw_image_dirs:
                if not os.path.exists(input_dir):
                    print(f"  [Warning] Directory not found: {input_dir}, skipping.")
                    continue

                sheet_name = os.path.basename(input_dir)
                
                files = []
                for ext in ["*.JPG", "*.jpg", "*.HEIC", "*.heic", "*.png"]:
                    files.extend(glob.glob(os.path.join(input_dir, ext)))
                
                print(f"  > Found {len(files)} images in '{sheet_name}'")

                for fpath in files:
                    fname = os.path.splitext(os.path.basename(fpath))[0]
                    
                    out_name = f"{sheet_name}_{fname}.jpg"
                    out_path = os.path.join(self.images_out_dir, out_name)
                    
                    tasks.append(executor.submit(self._convert_single_image, fpath, out_path))
                    
                    img_id = self.format_manager.extract_id(fname)
                    if img_id:
                        unique_id = f"{sheet_name}_{img_id}"
                        image_id_map[unique_id] = out_path

            count = 0
            for future in as_completed(tasks):
                if future.result():
                    count += 1
        
        print(f"  Done. Total mapped images: {len(image_id_map)}")
        return image_id_map

    def _convert_single_image(self, input_path, output_path):
        if os.path.exists(output_path):
            return True # Skip existing
        
        try:
            img = Image.open(input_path).convert("RGB")
            img.save(output_path, "JPEG", quality=95)
            return True
        except Exception as e:
            print(f"  [Error] Failed to convert {input_path}: {e}")
            return False

    def _process_metadata(self) -> pd.DataFrame:
        print(f"\n[2/3] Processing Metadata from {self.raw_metadata_file}...")
        
        df_dict = pd.read_excel(self.raw_metadata_file, header=1, sheet_name=None)
        
        processed_dfs = []
        for sheet_name, df in df_dict.items():
            print(f"  > Processing sheet: {sheet_name} (rows: {len(df)})")
            
            df = self._clean_dataframe(df, sheet_name)
            df = self._validate_values(df)
            
            processed_dfs.append(df)
            
        master_df = pd.concat(processed_dfs, ignore_index=True)
        master_df = self._validate_image_overlaps(master_df)
        self._show_label_distribution(master_df, self.required_columns[-1])

        print(f"  Done. Total valid rows: {len(master_df)}")
        return master_df

    def _clean_dataframe(self, df: pd.DataFrame, sheet_name):
        df.columns = df.columns.str.replace('\n', '')

        df = df.drop(index=0).reset_index(drop=True)
        df = df.rename(columns=self.rename_columns)
        df = df.dropna(subset=self.required_columns, how="any")

        df["sheet_name"] = sheet_name
        df["unique_patient_id"] = sheet_name + '_' + df["病患編號"].astype(str)
        
        return df

    def _validate_image_overlaps(self, df: pd.DataFrame) -> pd.DataFrame:
        print("    Validating image ID overlaps...")
        df_expanded = df.copy()
        
        df_expanded["expanded_ids"] = df_expanded["照片編號"].apply(self.format_manager.expand_ids)
        df_exploded = df_expanded.explode("expanded_ids")
        
        df_exploded["unique_img_key"] = (
            df_exploded["sheet_name"].astype(str) + "_" + df_exploded["expanded_ids"].astype(str)
        )
        
        overlap_check = df_exploded.groupby("unique_img_key")["unique_patient_id"].nunique()
        
        overlapping_imgs = overlap_check[overlap_check > 1].index.tolist()
        
        if not overlapping_imgs:
            return df
            
        print(f"    [Warning] Found {len(overlapping_imgs)} overlapping images across patients!")
        
        problematic_rows = df_exploded[df_exploded["unique_img_key"].isin(overlapping_imgs)]
        indices_to_drop = problematic_rows.index.unique()
        
        print("    [Error Details] The following rows have overlapping image IDs:")
        for idx in indices_to_drop:
            row = df.loc[idx]
            print(f"      - Row Index: {idx} | Sheet: {row['sheet_name']} | Patient: {row['病患編號']} | Photos: {row['照片編號']}")
            
        print(f"    [Action] Dropping {len(indices_to_drop)} rows to maintain data integrity.")
        
        return df.drop(index=indices_to_drop).reset_index(drop=True)

    def _validate_values(self, df: pd.DataFrame):
        df = df.copy()
        df["errors"] = ""

        # 身高異常 (1.0 ~ 2.0 m)
        height_mask = ~df["身高(m)"].between(1.0, 2.0)
        df.loc[height_mask, "errors"] += "身高異常 | "

        # 體重異常 (20 ~ 150 kg)
        weight_mask = ~df["體重(KG)"].between(20, 150)
        df.loc[weight_mask, "errors"] += "體重異常 | "

        # BMI異常 (10 ~ 50)
        bmi_mask = ~df["BMI"].between(10, 50)
        df.loc[bmi_mask, "errors"] += "BMI異常 | "

        # 術後天數為負
        days_mask = df["術後天數"] < 0
        df.loc[days_mask, "errors"] += "術後天數為負 |"

        # 傷口大小異常 (1 ~ 25)
        wound_size_mask = ~df["傷口大小"].between(1, 25)
        df.loc[wound_size_mask, "errors"] += "傷口大小異常 | "

        error_df = df[df["errors"] != ""]
        if not error_df.empty:
            print(f"    [Warning] Found {len(error_df)} rows with invalid values in this sheet.")
            print(error_df[["病患編號", "errors"]])

        df_clean = df[df["errors"] == ""]
        return df_clean

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

    def _save_results(self, df: pd.DataFrame, image_map):
        print(f"\n[3/3] Saving results...")
        
        df.to_csv(self.master_csv_path, index=False, encoding="utf-8-sig")
        
        with open(self.image_map_path, "w", encoding="utf-8") as f:
            json.dump(image_map, f, indent=4, ensure_ascii=False)
            
        print(f"  Master Metadata saved to: {self.master_csv_path}")
        print(f"  Image Map saved to: {self.image_map_path}")

if __name__ == "__main__":
    args = parse_args()

    raw_data_root       = args.raw_data_root
    raw_metadata_file   = os.path.join(raw_data_root, args.raw_metadata_file)
    raw_img_dirs        = [os.path.join(raw_data_root, dir_name) for dir_name in args.raw_img_dir_names]

    output_root         = args.output_root
    output_image_dir    = os.path.join(output_root, args.output_image_dirname)
    master_csv_path     = os.path.join(output_root, args.output_metadata_filename)
    img_map_file        = os.path.join(output_root, args.img_map_filename)

    builder = TKRDatasetBuilder(
        output_root=output_root,
        img_map_file=img_map_file,
        raw_image_dirs=raw_img_dirs,
        raw_metadata_file=raw_metadata_file,
        master_csv_path=master_csv_path,
        images_out_dir=output_image_dir,
    )
    builder.run()