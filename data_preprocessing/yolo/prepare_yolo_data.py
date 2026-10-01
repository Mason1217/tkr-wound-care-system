import os
import glob
import shutil
import random
import yaml
from PIL import Image, ImageOps
from tqdm import tqdm
from pillow_heif import register_heif_opener

# --- Settings ---
IMG_DIR = "data/TKR/processed/images"
ANNOTATION_DIR = "yolo/wound_dataset/yolo_annotation/labels"
OUTPUT_BASE_DIR = "yolo/wound_dataset"
DATA_YAML_FILE = "yolo/wound_data.yaml"
VAL_SIZE = 0.2
SEED = 42

def main():
    random.seed(SEED)
    register_heif_opener()

    print("正在搜尋原始圖片...")
    image_map = {}
    for ext in ["*.jpg", "*.JPG", "*.heic", "*.HEIC", "*.png"]:
        for img_path in glob.glob(os.path.join(IMG_DIR, ext)):
            processed_stem = os.path.splitext(os.path.basename(img_path))[0]
            stem = processed_stem[3:]
            image_map[stem] = img_path
    
    print(f"共找到 {len(image_map)} 張原始圖片。")

    label_files = glob.glob(os.path.join(ANNOTATION_DIR, "*.txt"))
    label_files = [f for f in label_files if "classes.txt" not in f]
    
    print(f"共找到 {len(label_files)} 個標註檔。")

    data_pairs = []
    for label_file in label_files:
        label_filename = os.path.basename(label_file)
        label_stem = os.path.splitext(label_filename)[0]
        
        # 解析檔名邏輯
        if "IMG_" in label_stem:
            img_name_start = label_stem.find("IMG_")
            original_img_name = label_stem[img_name_start:]
        elif "line_" in label_stem:
             img_name_start = label_stem.find("line_")
             original_img_name = label_stem[img_name_start:]
        else:
            original_img_name = label_stem

        if original_img_name in image_map:
            data_pairs.append({
                "label_src": label_file,
                "img_src": image_map[original_img_name],
                "target_stem": label_stem
            })
        else:
            print(f"[警告] {original_img_name} 找不到對應圖片: {label_filename}")

    print(f"成功配對 {len(data_pairs)} 組資料。")

    random.shuffle(data_pairs)
    split_idx = int(len(data_pairs) * (1 - VAL_SIZE))
    train_pairs = data_pairs[:split_idx]
    val_pairs = data_pairs[split_idx:]

    # 清空舊資料以避免混淆 (選擇性)
    # if os.path.exists(OUTPUT_BASE_DIR):
    #     shutil.rmtree(OUTPUT_BASE_DIR)

    for split_name, pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_dir = os.path.join(OUTPUT_BASE_DIR, "images", split_name)
        lbl_dir = os.path.join(OUTPUT_BASE_DIR, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        print(f"正在處理 {split_name} 集 ({len(pairs)} 筆)...")
        
        for pair in tqdm(pairs):
            shutil.copy(pair["label_src"], os.path.join(lbl_dir, pair["target_stem"] + ".txt"))
            
            img_src = pair["img_src"]
            img_dst = os.path.join(img_dir, pair["target_stem"] + ".jpg")
            
            try:
                img = Image.open(img_src)
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                img.save(img_dst, quality=95)
            except Exception as e:
                print(f"[錯誤] 處理圖片失敗 {img_src}: {e}")

    abs_path = os.path.abspath(OUTPUT_BASE_DIR)
    yaml_content = {
        "path": abs_path,
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["wound"]
    }
    
    with open(DATA_YAML_FILE, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    
    print(f"資料準備完成！請務必重新訓練模型。")

if __name__ == "__main__":
    main()