import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import cv2

from core.data.processor import TKRDataProcessor
from tkr_inference.heatmap import get_heatmap_visualizer

class Evaluator:
    def __init__(self, model: torch.nn.Module, dataloader, device, processor: TKRDataProcessor, model_cfg: DictConfig):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.processor = processor
        self.raw_results_df = None
        self.class_names = processor.classes_
        self.model_cfg = model_cfg

    def run_inference(self) -> pd.DataFrame:
        self.model.eval()
        self.model.to(self.device)

        results = []

        with torch.inference_mode():
            for batch in tqdm(self.dataloader, desc="Running Inference"):
                images, features, labels, metadata = batch
                images = images.to(self.device)
                features = features.to(self.device) if features.nelement() > 0 else None

                logits = self.model(images, features)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

                abnormal_probs = probs[:, 1] # Assuming class_1 is abnormal class in binary classification task
                labels_np = labels.cpu().numpy()
                
                for i in range(len(labels)):
                    p_id = metadata["patient_id"][i] if "patient_id" in metadata else ""
                    img_path = metadata["image_path"][i] if "image_path" in metadata else ""

                    results.append({
                        "patient_id": p_id,
                        "image_path": img_path,
                        "true_label": labels_np[i],
                        "abnormal_prob": abnormal_probs[i],
                    })
        self.raw_results_df = pd.DataFrame(results)
        return self.raw_results_df
    
    def generate_heatmaps(self, processed_dfs: dict, save_dir: str):

        img_df = processed_dfs.get("image_level_df")
        if img_df is None:
            print("[Warning] No image_level_df found. Please run apply_binary_rules() first.")
            return

        # build saving dirs
        heatmap_dir = os.path.join(save_dir, "heatmaps")
        categories = ["TP", "TN", "FP", "FN"]
        for cat in categories:
            os.makedirs(os.path.join(heatmap_dir, cat), exist_ok=True)

        # build mapping to get required information
        path_to_info = img_df.set_index("image_path")[["true_label", "pred_label", "abnormal_prob"]].to_dict('index')

        # initialize visualizer
        heatmap_viz = get_heatmap_visualizer(
            model_name=self.model_cfg.name,
            model=self.model,
            target_layer_name=self.model_cfg.get("target_layer", "layer4"),
        )
        
        # denormalize normalized images to original ones
        def denormalize(tensor):
            # use ImageNet settings as default
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = tensor.cpu().numpy().transpose(1, 2, 0)
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            return (img_np * 255).astype(np.uint8)

        self.model.eval()

        for batch in tqdm(self.dataloader, desc="Generating Heatmaps"):
            images, features, labels, metadata = batch
            images = images.to(self.device)
            
            for i in range(len(labels)):
                img_path = metadata["image_path"][i] if "image_path" in metadata else ""
                if not img_path or img_path not in path_to_info:
                    continue

                info = path_to_info[img_path]
                true_lbl = int(info["true_label"])
                pred_lbl = int(info["pred_label"])

                # classify to corresponding category
                if true_lbl == 1 and pred_lbl == 1:
                    category = "TP"
                elif true_lbl == 0 and pred_lbl == 0:
                    category = "TN"
                elif true_lbl == 0 and pred_lbl == 1:
                    category = "FP"
                else:
                    category = "FN"

                single_img_tensor = images[i].unsqueeze(0).requires_grad_(True)
                orig_img_np = denormalize(images[i])
                heatmap_img = heatmap_viz.generate_heatmap(single_img_tensor, orig_img_np)

                # setup saving format
                filename = os.path.splitext(os.path.basename(img_path))[0] + "_heatmap.png"
                save_path = os.path.join(heatmap_dir, category, f"{category}_{filename}")
                
                # plot and save
                self._plot_heatmap(orig_img_np, heatmap_img, info, save_path)
                
    def _plot_heatmap(self, original_img: np.ndarray, heatmap: np.ndarray, result: dict, save_path: str):

        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Microsoft JhengHei", "SimHei", "WenQuanYi Zen Hei"] 
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        true_label = result["true_label"]
        pred_label = result["pred_label"]
        abnormal_prob = result["abnormal_prob"]

        is_correct = (true_label == pred_label)
        title_color = "green" if is_correct else "red"

        title_str = (
            f"File: {os.path.basename(save_path)}\n"
            f"Ground Truth: {true_label} | Predict: {pred_label} | "
            f"Abnormal Probability: {abnormal_prob:.2f}"
        )
        fig.suptitle(title_str, fontsize=14, color=title_color, fontweight="bold")

        axes[0].imshow(heatmap)
        axes[0].axis("off")
        axes[0].set_title("Heatmap", fontsize=12)

        axes[1].imshow(original_img)
        axes[1].axis("off")
        axes[1].set_title("Original", fontsize=12)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    
    def apply_binary_rules(self, threshold: float = 0.5, agg_patient: bool = False) -> dict:
        """
        Note: **Assuming class 1 is the abnormal class**

        Args:
            threshold (float): for abnormal class
            agg_patient (bool): default using max risk voting
        """
        if self.raw_results_df is None:
            raise ValueError("Must run_inference() first.")
        
        df = self.raw_results_df.copy()
        df["pred_label"] = (df["abnormal_prob"] >= threshold).astype(int)
        df["is_correct"] = (df["pred_label"] == df["true_label"])

        if agg_patient:
            patient_df = df.groupby("patient_id").agg(
                true_patient_label=("true_label", "max"),
                max_abnormal_prob=("abnormal_prob", "max"),
                image_count=("image_path", "count"),
                pred_sum=("pred_label", "sum"),
            ).reset_index()

            patient_df["patient_pred_label"] = (patient_df["max_abnormal_prob"] >= threshold).astype(int)
            patient_df["is_unstable"] = (patient_df["pred_sum"] > 0) & (patient_df["pred_sum"] < patient_df["image_count"])

        return {
            "image_level_df": df,
            "patient_level_df": patient_df,
        }
    
    def export_reports(self, processed_dfs: dict, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        img_df = processed_dfs.get("image_level_df")
        pat_df = processed_dfs.get("patient_level_df", None)
        
        self._export_reports_img_level(img_df, save_dir)

        if pat_df is not None:
            self._export_reports_pat_level(pat_df, img_df, save_dir)
            
    def _export_reports_img_level(self, img_df: pd.DataFrame, save_dir):

        self._classification_report(
            img_df["true_label"],
            img_df["pred_label"],
            save_dir,
            "classification_report_image_level.txt",
        )
        
        img_df.to_csv(os.path.join(save_dir, "inference_results_image_level.csv"), index=False)

        high_conf_errors = img_df[
            (img_df["is_correct"] == False) &
            ((img_df["abnormal_prob"] >= 0.8) | (img_df["abnormal_prob"] <= 0.2))
        ]
        high_conf_errors.to_csv(os.path.join(save_dir, "high_confidence_errors.csv"), index=False)
    
    def _export_reports_pat_level(self, pat_df: pd.DataFrame, img_df: pd.DataFrame, save_dir):

        pat_df.to_csv(os.path.join(save_dir, "patient_level_voting_results.csv"), index=False)

        self._classification_report(
            pat_df["true_patient_label"],
            pat_df["patient_pred_label"],
            save_dir,
            "classification_report_patient_level.txt",
        )

        unstable_patient_ids = pat_df[pat_df["is_unstable"]]["patient_id"]

        if not unstable_patient_ids.empty:
            unstable_records = img_df[img_df["patient_id"].isin(unstable_patient_ids)]
            unstable_records = unstable_records.sort_values(by=["patient_id", "image_path"])
            unstable_records.to_csv(os.path.join(save_dir, "unstable_patients_details.csv"), index=False)

    def _classification_report(self, true, pred, save_dir, filename):

        report_str = classification_report(true, pred, target_names=self.class_names, digits=4, zero_division=0,)
        
        with open(os.path.join(save_dir, filename), "w") as f:
            f.write(f"=== {filename} ===\n")
            f.write(report_str)

        import json
        report_dict = classification_report(true, pred, target_names=self.class_names, digits=4, zero_division=0, output_dict=True)
        json_filename = filename.replace(".txt", ".json")
        
        with open(os.path.join(save_dir, json_filename), "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=4, ensure_ascii=False)
    
    def get_cm(self, processed_dfs: dict):
        '''
        Returns:
            tuple: (img_cm, pat_cm)
        '''
        img_df = processed_dfs.get("image_level_df")
        pat_df = processed_dfs.get("patient_level_df", None)

        img_cm = self._confusion_mat(img_df)
        pat_cm = self._confusion_mat(pat_df)

        return img_cm, pat_cm

    def _confusion_mat(self, df: pd.DataFrame):
        pass