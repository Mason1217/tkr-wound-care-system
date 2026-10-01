import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, recall_score, fbeta_score, accuracy_score, f1_score
import torch
import time
from PIL import Image
from functools import wraps

plt.rcParams['font.sans-serif'] = ['PingFang TC', 'Heiti TC', 'Arial Unicode MS']

from inference.pipeline.pipeline import TKRInferencePipeline
from .heatmap import BaseHeatmapVisualizer
from .reliability_diagram import plot_reliability_diagram
from .auc_pr import auc_pr
from .auc_roc import auc_roc
from .bootstrap_ci import bootstrap_ci
from .tsne import tsne_analysis_grid_sweep

HEIGHT_COL   = "身高(m)"
WEIGHT_COL   = "體重(KG)"
SEXUAL_COL   = "性別"
BMI_COL      = "BMI"
POST_OP_COL  = "術後天數"
WOUND_SZ_COL = "傷口大小"
LABEL_COL    = "建議"

LABEL_MAPPING = {
    "正常，冰敷即可" : 0,
    "低風險，加強冰敷" : 0,
    "中度風險，觀察3天": 1,
    "高風險，立即就醫": 1,
}

TARGET_NAMES = ["Normal sample", "Abnormal sample"]

def profile_end_to_end(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        device = self.pipeline.device

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        start_time = time.perf_counter()

        result_dict = func(self, *args, **kwargs)

        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        elif device.type == "mps":
            torch.mps.synchronize()
            peak_mem = torch.mps.current_allocated_memory() / (1024 ** 2)
        else:
            peak_mem = 0.0

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        if isinstance(result_dict, dict) and result_dict.get("status") != "SKIP":
            result_dict["e2e_latency_ms"] = latency_ms
            result_dict["e2e_peak_mem_mb"] = peak_mem

        return result_dict

    return wrapper

class TKRSystemEvaluator:
    def __init__(self, pipeline: TKRInferencePipeline, test_csv_path: str):

        self.pipeline = pipeline
        self.test_df  = pd.read_csv(test_csv_path)
        print(f"[System Evaluator]\t Load testing dataframe from {test_csv_path}")

    def evaluate(self, save_dir: str, reliability_diagram_filename: str | None = None, auc_pr_filename: str | None = None, auc_roc_filename: str | None = None, tsne_filename: str | None = None, tsne_perplexities: list[int] | None = None) -> list[dict]:

        results = []
        y_true  = []
        y_pred  = []
        y_prob  = []
        abnormal_prob = []
        fold_embeddings = None
        fold_preds      = None
        tsne_y_true     = []
        e2e_latencies = []
        e2e_memories  = []

        print( "[System Evaluator]\t Starting Evaluation...")
        print(f"[System Evaluator]\t Testing dataframe size {len(self.test_df)}")

        for idx, row in tqdm(self.test_df.iterrows(), total=len(self.test_df)):

            res_dict = self._process_single_row(row)

            if res_dict.get("status") == "SKIP":
                print(f"[Evaluator]\t [Warning]\t {idx}th row skipped: {res_dict['error_msg']}")
                continue
                
            results.append(res_dict)

            if res_dict["status"] == "SUCCESS":
                y_true.append(res_dict["true_label"])
                y_pred.append(res_dict["pred_label"])
                y_prob.append(res_dict["probability"])
                abnormal_prob.append(res_dict["abnormal_prob"])

                if res_dict.get("fold_embeddings") is not None:
                    entries = res_dict["fold_embeddings"]
                    if fold_embeddings is None:
                        fold_embeddings = [[] for _ in entries]
                        fold_preds      = [[] for _ in entries]
                    for k, entry in enumerate(entries):
                        fold_embeddings[k].append(entry["embedding"])
                        fold_preds[k].append(entry["pred_label"])
                    tsne_y_true.append(res_dict["true_label"])

                if "e2e_latency_ms" in res_dict:
                    e2e_latencies.append(res_dict["e2e_latency_ms"])
                    e2e_memories.append(res_dict["e2e_peak_mem_mb"])

        if e2e_latencies:
            print("\n[System Evaluator]\t --- End-to-End Profiling Summary ---")
            print( f"[System Evaluator]\t Average E2E Latency : {np.mean(e2e_latencies):.2f} ms")
            print( f"[System Evaluator]\t P95 E2E Latency     : {np.percentile(e2e_latencies, 95):.2f} ms")
            print( f"[System Evaluator]\t Throughput          : {1000 / np.mean(e2e_latencies):.2f} images / sec")
            print( f"[System Evaluator]\t Average Peak Memory : {np.mean(e2e_memories):.2f} MB")
            print("-" * 60, "\n")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        self.export_results(results, save_dir)
        self.export_image_level_classification_report(y_true, y_pred, TARGET_NAMES, save_dir)

        if self.pipeline.heatmap_viz is not None:
            self.visualize_heatmap(self.pipeline.heatmap_viz, results, save_dir)

        if reliability_diagram_filename is not None:
            reliability_diagram_path = os.path.join(save_dir, reliability_diagram_filename)
            plot_reliability_diagram(y_true, y_pred, y_prob, save_path=reliability_diagram_path)

        if auc_pr_filename is not None:
            auc_pr_path = os.path.join(save_dir, auc_pr_filename)
            auc_pr(y_true, abnormal_prob, auc_pr_path)

        if auc_roc_filename is not None:
            auc_roc_path = os.path.join(save_dir, auc_roc_filename)
            auc_roc(y_true, abnormal_prob, auc_roc_path)

        if tsne_filename is not None and fold_embeddings:
            tsne_path = os.path.join(save_dir, tsne_filename)
            perplexities = tsne_perplexities if tsne_perplexities else [30]
            tsne_analysis_grid_sweep(
                [np.array(fe) for fe in fold_embeddings], tsne_y_true, TARGET_NAMES,
                y_pred_list=fold_preds, save_path=tsne_path, perplexities=perplexities,
            )

        return results

    @profile_end_to_end
    def _process_single_row(self, row: pd.Series) -> dict:
            image_path = row["image_path"]
            label_col = row.get(LABEL_COL)
            
            if label_col is None:
                return {"status": "SKIP", "error_msg": "No label column"}
                
            true_label = LABEL_MAPPING.get(label_col)
            if true_label is None:
                return {"status": "SKIP", "error_msg": f"Unknown label: {label_col}"}

            patient_data = {
                "BMI" : row.get(BMI_COL, None),
                "術後天數" : row.get(POST_OP_COL, None),
                "傷口大小" : row.get(WOUND_SZ_COL, None),
            }

            result = {
                "image_path": image_path,
                "true_label": true_label,
                "pred_label": -1,
                "probability": None,
                "abnormal_prob": None,
                "inference_mode": "FAILED",
                "status": "FAILED",
                "error_msg": "",
                "fold_embeddings": None,
            }

            try:
                pred_dict = self.pipeline.predict(image_path, patient_data)
                result.update({
                    "pred_label": pred_dict["is_abnormal"],
                    "probability": pred_dict["probability"],
                    "abnormal_prob": pred_dict["abnormal_probability"],
                    "inference_mode": pred_dict["inference_mode"],
                    "fold_embeddings": pred_dict.get("fold_embeddings"),
                    "status": "SUCCESS"
                })
            except ValueError as e:
                result["error_msg"] = str(e)
                if str(e) == "WOUND_NOT_FOUND":
                    print(f"[System Evaluator]\t [Error]\t {e} at {image_path}")
                elif str(e) == "IMAGE_LOAD_FAILED":
                    print(f"[Run Inference]\t [Warning]\t Cannot load image at {image_path}")
                else:
                    raise e
            except Exception as e:
                result["error_msg"] = str(e)
                print(f"[System Evaluator]\t [Error]\t {e}")

            return result

    def export_results(self, results: list[dict], save_dir: str):

        result_df = pd.DataFrame(results).drop(columns=["fold_embeddings"], errors="ignore")
        result_df.to_csv(os.path.join(save_dir, "evaluation_details.csv"))
    
    def export_image_level_classification_report(self, y_true: list, y_pred: list, target_names: list, save_dir: str):
        
        target_names_lower = [name.lower() for name in target_names]
        normal_idx = 0
        abnormal_idx = 1
        labels = range(len(target_names))

        precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        f2_scores               = fbeta_score(y_true, y_pred, beta=2, labels=labels, average=None, zero_division=0)
        
        abnormal_recall     = recall[abnormal_idx]
        abnormal_f2         = f2_scores[abnormal_idx]
        abnormal_precision  = precision[abnormal_idx]
        normal_recall       = recall[normal_idx]

        # Abnormal metrics
        mean_recall, lower_recall, upper_recall = bootstrap_ci(y_true, y_pred, recall_score, pos_label=1)
        mean_f2score, lower_f2score, upper_f2score = bootstrap_ci(y_true, y_pred, fbeta_score, beta=2, pos_label=1)
        
        acc         = accuracy_score(y_true, y_pred)
        macro_f1    = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        report_str = (
            f"=== Custom Evaluation Report ===\n"
            f"Abnormal Recall    : {mean_recall:.4f} (95% CI: {lower_recall:.4f} - {upper_recall:.4f})\n"
            f"Abnormal F2-score  : {mean_f2score:.4f} (95% CI: {lower_f2score:.4f} - {upper_f2score:.4f})\n"
            f"Abnormal Precision : {abnormal_precision:.4f}\n"
            f"Normal Recall      : {normal_recall:.4f}\n"
            f"Accuracy           : {acc:.4f}\n"
            f"Macro F1-score     : {macro_f1:.4f}\n"
        )
        
        with open(os.path.join(save_dir, "classification_report_image_level.txt"), "w") as f:
            f.write(report_str)
        
        return {
            "Abnormal Recall": abnormal_recall,
            "Abnormal F2-score": abnormal_f2,
            "Abnormal Precision": abnormal_precision,
            "Normal Recall": normal_recall,
            "Accuracy": acc,
            "Macro F1-score": macro_f1
        }

    def visualize_heatmap(self, visualizer: BaseHeatmapVisualizer, results: list[dict], save_dir: str):

        heatmap_dir = os.path.join(save_dir, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)

        for result in tqdm(results, total=len(results)):

            if result.get("status") == "FAILED" or result.get("abnormal_prob") is None:
                print(f"[System Evaluator]\t [Warning]\t Skip {result['image_path']}")
                continue

            image_path = result["image_path"]

            if self.pipeline.cropper is not None:
                original_img = np.array(self.pipeline.cropper.crop_single_image(image_path))
            else:
                original_img = np.array(cv2.imread(image_path))

            img_tensor = self.pipeline.preprocess_image(
                self.pipeline.detector,
                self.pipeline.cropper,
                self.pipeline.image_transforms,
                image_path,
                self.pipeline.device,
            )

            heatmap = visualizer.generate_heatmap(img_tensor, original_img)
            self._plot_heatmap(heatmap, original_img, result, heatmap_dir)

    def _plot_heatmap(self, heatmap: np.ndarray, original_img: np.ndarray, result: dict, heatmap_dir: str):

        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Microsoft JhengHei", "SimHei", "WenQuanYi Zen Hei"] 
        plt.rcParams['axes.unicode_minus'] = False

        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        true_label = result["true_label"]
        pred_label = result["pred_label"]
        abnormal_prob = result["abnormal_prob"]

        is_correct = (true_label == pred_label)
        title_color = "green" if is_correct else "red"
        filename = os.path.basename(result["image_path"])

        title_str = (
            f"File: {filename}\n"
            f"Ground Truth: {true_label} | Predict: {pred_label} | "
            f"Abnormal Probability: {abnormal_prob:.2f}"
        )
        fig.suptitle(title_str, fontsize=14, color=title_color, fontweight="bold")

        axes[0].imshow(heatmap)
        axes[0].axis("off")
        axes[0].set_title("Heatmap", fontsize=12)

        axes[1].imshow(original_rgb)
        axes[1].axis("off")
        axes[1].set_title("Original", fontsize=12)

        plt.tight_layout()

        save_path = os.path.join(heatmap_dir, f"heatmap_{filename}")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def profile_model(self, num_runs: int = 100) -> dict:
        '''
        Get model's number of parameters, inference latency, throughput, and memory (optional)

        Returns:
            results(dict):
                {                                               <br>
                 'total_param'    ,                             <br>
                 'trainable_param',                             <br>
                 'ave_latency', **(ms)**                        <br>
                 'p95_latency', **(95% of inference latency)**  <br>
                 'throughput' , **(images / sec)**              <br>
                 'max_mem'.   , **(optional, measured in MB)**  <br>
                }
        '''
        device    = self.pipeline.device
        print(f"[System Evaluator]\t Profiling model on device: {device}")

        img_tensor      = self.pipeline.image_transforms(Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)))
        img_tensor      = img_tensor.unsqueeze(0).to(device)
        tabular_tensor  = torch.randn(1, 3, dtype=torch.float32).to(device)

        latencies = []
        result    = {}
        
        if self.pipeline.multimodal_ensemble is not None  : model = self.pipeline.multimodal_ensemble.models[0]
        elif self.pipeline.image_only_ensemble is not None: model = self.pipeline.image_only_ensemble.models[0]
        else: raise ValueError(f"[System Evaluator]\t [Error]\t pipeline has no model for profiling.")

        result["total_param"]     = sum(p.numel() for p in model.parameters())
        result["trainable_param"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Warming up for 10 runs
        with torch.inference_mode():
            for _ in tqdm(range(10), total=10, desc="warming up"):
                _ = model(img_tensor, tabular_tensor)
        
        # Measure inference latency
        with torch.inference_mode():
            for _ in tqdm(range(num_runs), total=num_runs, desc="measuring latency"):
                
                if device.type == "cuda": torch.cuda.synchronize()
                elif device.type == "mps": torch.mps.synchronize()
                start_time = time.perf_counter()
                
                _ = model(img_tensor, tabular_tensor)

                if device.type == "cuda": torch.cuda.synchronize()
                elif device.type == "mps": torch.mps.synchronize()
                end_time = time.perf_counter()

                latencies.append((end_time - start_time) * 1000) # convert to ms
        
        result["ave_latency"] = np.mean(latencies)
        result["p95_latency"] = np.percentile(latencies, 95)
        result["throughput"]  = 1000 / result["ave_latency"] # image / sec

        # Measure maximum memory allocated
        max_mem = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                _ = model(img_tensor, tabular_tensor)
            max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # measured in MB
        
        elif device.type == "mps":
            with torch.inference_mode():
                _ = model(img_tensor, tabular_tensor)
            max_mem = torch.mps.current_allocated_memory() / (1024 **2)
            
        result["max_mem"] = max_mem

        print(f"[System Evaluator]\t Total parameters    \t: {result.get('total_param', 0) / 1e6:.2f} M")
        print(f"[System Evaluator]\t Trainable parameters\t: {result.get('trainable_param', 0) / 1e6:.2f} M")
        print(f"[System Evaluator]\t Average latency     \t: {result.get('ave_latency', 0):.2f} ms")
        print(f"[System Evaluator]\t P95 latency         \t: {result.get('p95_latency', 0):.2f} ms")
        print(f"[System Evaluator]\t Throughput          \t: {result.get('throughput', 0):.2f} images / sec")
        print(f"[System Evaluator]\t Peak VRAM Usage     \t: {result.get('max_mem', 0):.2f} MB")
        
        return result
