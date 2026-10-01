import os
import argparse
import pandas as pd

EXUDATE     = "滲液"
SWELLEXT    = "紅腫程度"
CRACK       = "裂開"
LABEL       = "建議"

def parse_args():
    parser = argparse.ArgumentParser(description="Clean wound assessment metadata.")
    parser.add_argument("--drop_sus", action="store_true", required=True, help="Drop suspicious samples")
    return parser.parse_args()

def check_label_logic(csv_path):
    args = parse_args()
    df = pd.read_csv(csv_path)
    
    normal_labels = ["正常，冰敷即可", "低風險，加強冰敷"]
    abnormal_labels = ["中度風險，觀察3天", "高風險，立即就醫"]
    
    mask_severe_symptoms = (
        df[EXUDATE].isin(["有-暗紅褐色", "有-鮮血"]) |
        (df[SWELLEXT] == "深紅，大範圍") |
        (df[CRACK] == "有")
    )
    mask_labeled_normal = df[LABEL].isin(normal_labels)
    
    suspicious_fn = df[mask_severe_symptoms & mask_labeled_normal]
    
    mask_no_symptoms = (
        (df[EXUDATE] == "無") &
        (df[SWELLEXT] == "無") &
        (df[CRACK] == "無")
    )
    mask_labeled_abnormal = df[LABEL].isin(abnormal_labels)
    
    suspicious_fp = df[mask_no_symptoms & mask_labeled_abnormal]
    
    print(f"Found {len(suspicious_fn)} UNDERESTIMATED samples.")
    print(f"Found {len(suspicious_fp)} OVERESTIMATED samples.")
    
    suspicious_all = pd.concat([suspicious_fn, suspicious_fp])
    suspicious_all.to_csv("suspicious_labels.csv", index=False, encoding="utf-8-sig")

    if args.drop_sus:
        clean_df = df.drop(index=suspicious_all.index)
        base, ext = os.path.splitext(csv_path)
        output_path = f"{base}_clean{ext}"

        clean_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Clean metadata saved to {output_path}")
        print(f"Original records: {len(df)} -> Cleaned records: {len(clean_df)}")

if __name__ == "__main__":
    check_label_logic("data/TKR/processed/master_metadata.csv")