import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch

from utils.gradcam import GradCAM

class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Microsoft JhengHei", "SimHei", "WenQuanYi Zen Hei"] 
        plt.rcParams['axes.unicode_minus'] = False

    def plot_confusion_matrix(self, cm, class_names, filename="confusion_matrix.png"):
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved CM to {save_path}")

    def plot_loss_curves(self, history_csv_path: str):
        """
        Read history.csv and plot train/val loss curves
        """
        if not os.path.exists(history_csv_path):
            print(f"File not found: {history_csv_path}")
            return
            
        df = pd.read_csv(history_csv_path)
        
        plt.figure(figsize=(10, 6))
        
        # Train loss
        if "train_loss" in df.columns:
            plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker='.')
            
        # Val loss
        if "val_loss" in df.columns:
            plt.plot(df["epoch"], df["val_loss"], label="Val Loss", marker='.')
            
        plt.title(f"Loss Curve ({os.path.basename(history_csv_path)})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        save_name = os.path.basename(history_csv_path).replace(".csv", ".png")
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Loss curve saved to {save_path}")

    def plot_model_focus(
        self, 
        model: torch.nn.Module, 
        dataloader, 
        device, 
        mean, 
        std, 
        method="gradcam", 
        errors_only=False
    ):
        """
        統一的模型關注點視覺化管線。
        method: 'gradcam' 或 'attention'
        """
        viz_dir = os.path.join(self.output_dir, f"{method}_maps{' _errors' if errors_only else ''}")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"Generating {method} maps to: {viz_dir}")

        model.eval()
        model.to(device)

        # 針對 Grad-CAM 的初始化
        grad_cam_extractor = None
        if method == "gradcam":
            target_layer = self._get_target_layer(model.backbone)
            grad_cam_extractor = GradCAM(model, target_layer)

        count = 0
        mean_arr = np.array(mean)
        std_arr = np.array(std)

        # Grad-CAM 需要計算梯度，因此不能全域使用 inference_mode
        context_manager = torch.inference_mode() if method == "attention" else torch.enable_grad()

        with context_manager:
            for batch in tqdm(dataloader, desc=f"Generating {method}", leave=False):
                # 兼容不同 dataloader 輸出格式 (有無 features)
                if len(batch) == 4:
                    images, features, labels, sample_ids = batch
                    features = features.to(device)
                else:
                    images, labels, sample_ids = batch
                    features = None
                
                images = images.to(device)

                # 1. 取得預測結果
                if features is not None:
                    logits = model(images, features)
                else:
                    logits = model(images)
                preds = torch.argmax(logits, dim=1)

                # 若為 Attention，整批提取以節省效能
                batch_attentions = None
                if method == "attention":
                    outputs = model.backbone(pixel_values=images, output_attentions=True)
                    batch_attentions = outputs.attentions[-1].detach().cpu()

                # 2. 逐張圖片處理
                for i in range(images.size(0)):
                    
                    pred_val = preds[i].item()
                    label_val = labels[i].item()

                    if errors_only and pred_val == label_val:
                        continue

                    img_tensor = images[i].unsqueeze(0)
                    img_path = sample_ids["image_path"][i] if isinstance(sample_ids, dict) else sample_ids[i]
                    
                    # 3. 計算 Heatmap (0~1 之間的 2D Numpy Array)
                    if method == "gradcam":
                        # Grad-CAM 內部執行 backward，需確保 requires_grad 狀態
                        heatmap = self._generate_gradcam_mask(grad_cam_extractor, img_tensor)
                    elif method == "attention":
                        heatmap = self._generate_attention_mask(batch_attentions[i])
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    # 4. 繪製並存檔
                    filename = f"P{pred_val}_T{label_val}_{os.path.basename(img_path)}"
                    save_path = os.path.join(viz_dir, filename)
                    
                    self._save_unified_overlay_plot(
                        image_tensor=images[i],
                        heatmap_mask=heatmap,
                        pred=pred_val,
                        label=label_val,
                        save_path=save_path,
                        mean=mean_arr,
                        std=std_arr,
                        original_file=img_path,
                        title_suffix=method.upper()
                    )
                    count += 1

    # =========================================================
    #  Algorithm Specific Extractors
    # =========================================================

    def _get_target_layer(self, backbone):
        if hasattr(backbone, 'layer4'):
            return backbone.layer4[-1]
        elif hasattr(backbone, 'blocks'):
            return backbone.blocks[-1]
        raise AttributeError(f"Cannot find target layer for Grad-CAM in backbone: {type(backbone).__name__}")

    def _generate_gradcam_mask(self, grad_cam, img_tensor):
        heatmap, _ = grad_cam(img_tensor)
        return heatmap # 假設你外部的 GradCAM 類別回傳的是 0-1 正規化後的 2D numpy array

    def _generate_attention_mask(self, attention_tensor, target_size=(224, 224)):
        avg_heads_att = attention_tensor.mean(dim=0) 
        cls_to_patch_att = avg_heads_att[0, 1:] 
        num_patches_side = int(cls_to_patch_att.shape[0]**0.5)
        
        mask = cls_to_patch_att.reshape(num_patches_side, num_patches_side).numpy()
        mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_CUBIC)
        
        min_val, max_val = mask_resized.min(), mask_resized.max()
        if max_val - min_val > 1e-5:
            mask_resized = (mask_resized - min_val) / (max_val - min_val)
        else:
            mask_resized = np.zeros_like(mask_resized)
            
        return mask_resized

    # =========================================================
    #  Unified Plotting Tools
    # =========================================================

    def _unnormalize_image(self, tensor, mean, std):
        """將 Tensor 轉回 0-1 的 Numpy 陣列供 Matplotlib 顯示"""
        img_np = tensor.cpu().permute(1, 2, 0).numpy()
        img_np = std * img_np + mean
        return np.clip(img_np, 0, 1)

    def _save_unified_overlay_plot(self, image_tensor, heatmap_mask, pred, label, save_path, mean, std, original_file, title_suffix):
        """
        統一的繪圖函數：左側原圖，右側疊加熱圖。
        標題包含檔名、預測與真實標籤，正確為綠色，錯誤為紅色。
        """
        # 1. 還原原始圖片
        img_to_plot = self._unnormalize_image(image_tensor, mean, std)
        
        # 2. 調整熱圖尺寸以符合原圖
        h, w = img_to_plot.shape[:2]
        if heatmap_mask.shape != (h, w):
            heatmap_mask = cv2.resize(heatmap_mask, (w, h))
        
        # 3. 設定畫布
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # 標題設定
        is_correct = (pred == label)
        title_color = 'green' if is_correct else 'red'
        file_name = os.path.basename(original_file)
        fig.suptitle(f"Prediction: {pred} | Ground Truth: {label}\nFile: {file_name}", 
                     fontsize=12, color=title_color, fontweight='bold')
        
        # 左側：原始圖片
        axes[0].imshow(img_to_plot)
        axes[0].axis('off')
        axes[0].set_title("Original Image")
        
        # 右側：疊加熱圖
        axes[1].imshow(img_to_plot)
        heatmap_img = axes[1].imshow(heatmap_mask, cmap='jet', alpha=0.5, vmin=0, vmax=1) 
        axes[1].axis('off')
        axes[1].set_title(f"Overlay ({title_suffix})")
        
        # 加入 Color Bar 供右側熱圖參考
        cbar = fig.colorbar(heatmap_img, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_ticks([0, 0.5, 1]) 
        cbar.set_ticklabels(['Low', 'Med', 'High'])
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)