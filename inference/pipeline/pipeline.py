import cv2
from PIL import Image
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig

from data_preprocessing.cropper.base_cropper import BaseWoundCropper
from core.data.processor import TKRDataProcessor
from core.data.transforms import build_transforms

from inference.detector import BaseWoundDetector
from inference.evaluate.heatmap import BaseHeatmapVisualizer

class TKRInferencePipeline:
    '''
    Note: for binary task with **label0:** normal & **label1:** abnormal
    '''
    def __init__(
            self,
            multimodal_ensemble : nn.Module | None,
            image_only_ensemble : nn.Module | None,
            detector            : BaseWoundDetector | None,
            cropper             : BaseWoundCropper  | None,
            tabular_processors  : list[TKRDataProcessor],
            image_transforms    : transforms.Compose,
            calib_params        : DictConfig,
            device              : str,
            image_only          : bool = False,
            heatmap_viz         : BaseHeatmapVisualizer | None = None,
            extract_embeddings  : bool = False,
    ):
        self.detector            = detector
        self.cropper             = cropper
        self.tabular_processors  = tabular_processors
        self.image_transforms    = image_transforms
        self.mm_calib            = calib_params.multimodal
        self.im_calib            = calib_params.image_only
        self.device              = torch.device(self._resolve_device(device))
        self.image_only          = image_only
        self.heatmap_viz         = heatmap_viz
        self.extract_embeddings  = extract_embeddings

        self.multimodal_ensemble = multimodal_ensemble.to(self.device).eval() if multimodal_ensemble is not None else None
        self.image_only_ensemble = image_only_ensemble.to(self.device).eval() if image_only_ensemble is not None else None

        self.required_keys       = ["BMI", "術後天數", "傷口大小"]
        self.abnormal_suggestion = "異常需醫療介入"
        self.normal_suggestion   = "觀察和冰敷即可"
        self.abnormal_idx        = 1

        print(f"[Pipeline]\t image only mode: {self.image_only}")
        print(f"[Pipeline]\t device: {self.device}")
        print(f"[Pipeline]\t multimodal temperatures: {self.mm_calib.temperatures} | threshold: {self.mm_calib.threshold}")
        print(f"[Pipeline]\t image_only temperatures: {self.im_calib.temperatures} | threshold: {self.im_calib.threshold}")
    
    @staticmethod
    def preprocess_image(detector: BaseWoundDetector | None, cropper: BaseWoundCropper | None, image_transforms: transforms.Compose, image_path: str, device: torch.device) -> torch.Tensor:
        '''
        YOLO cropping -> RGB -> PIL -> transforms -> tensor
        '''
        if detector is not None and not detector.has_wound(image_path):
            raise ValueError("WOUND_NOT_FOUND")
        
        if cropper is not None:
            img_array = cropper.crop_single_image(image_path)
            if img_array is None:
                raise ValueError("WOUND_NOT_FOUND")            
        else:
            img_array = cv2.imread(image_path)
            if img_array is None:
                raise ValueError("IMAGE_LOAD_FAILED")
        
        img = Image.fromarray(
            cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        )
        
        transformed = image_transforms(img)

        if not isinstance(transformed, torch.Tensor): raise TypeError("[Pipeline]\t [Error]\t Transforms pipeline did not return a torch.Tensor.")
        return transformed.unsqueeze(0).to(device)

    @staticmethod
    def preprocess_tabular(processor: TKRDataProcessor, patient_data: dict, device: torch.device) -> torch.Tensor:
        '''
        Dict -> series -> processor -> tensor
        '''
        series = pd.Series(patient_data)
        feat_tensor = processor.process_features(series)
        return feat_tensor.unsqueeze(0).to(device)

    @staticmethod
    def _resize_to_height(img: np.ndarray, target_height: int) -> np.ndarray | None:
        if img is None or img.ndim != 3 or target_height <= 0:
            return None

        height, width = img.shape[:2]
        if height <= 0 or width <= 0:
            return None
        if height == target_height:
            return img

        target_width = max(1, int(round(width * (target_height / height))))
        return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

    @classmethod
    def _hstack_same_height(cls, left: np.ndarray | None, right: np.ndarray | None) -> np.ndarray | None:
        if left is None or right is None:
            return None
        if left.ndim != 3 or right.ndim != 3:
            return None

        target_height = min(left.shape[0], right.shape[0])
        left_resized = cls._resize_to_height(left, target_height)
        right_resized = cls._resize_to_height(right, target_height)

        if left_resized is None or right_resized is None:
            return None

        return np.hstack((left_resized, right_resized))

    def predict(self, raw_image_path: str, patient_data: dict) -> dict:
        '''
        Args:
            raw_image_path(str): path to raw image
            patient_data(dict) : {"BMI": , "術後天數": , "傷口大小": }
        
        Returns:
            result(dict): {
                **"is_abnormal"**: bool,
                **"abnormal_probability"**: float,
                **"probability"**: float,
                **"inference_mode"**: multimodel/image_only,
                **"suggestion"**: 異常需醫療介入/觀察和冰敷即可,
                **"fold_embeddings"**: list[dict] | None, one {"embedding", "pred_label"} per fold (only when extract_embeddings=True)
            }
        
        Exceptions:
            ValueError:
                WOUND_NOT_FOUND: raised when yolo doesn't detect TKR wound
        '''
        
        img_tensor = self.preprocess_image(self.detector, self.cropper, self.image_transforms, raw_image_path, self.device)

        probs, used_mode, current_threshold = self._forward(img_tensor, patient_data)
        abnormal_prob, is_abnormal = self._evaluate(probs, current_threshold)

        heatmap = None
        if self.heatmap_viz is not None:

            if self.cropper is not None:
                original_img = self.cropper.crop_single_image(raw_image_path)
            else:
                original_img = cv2.imread(raw_image_path)

            if original_img is not None:
                try:
                    generated_heatmap = self.heatmap_viz.generate_heatmap(img_tensor, original_img)
                    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    heatmap = self._hstack_same_height(generated_heatmap, original_rgb)
                    if heatmap is None:
                        print(
                            "[Pipeline]\t [Warning]\t Heatmap generation returned invalid shapes "
                            f"heatmap={getattr(generated_heatmap, 'shape', None)} "
                            f"original={getattr(original_rgb, 'shape', None)}"
                        )
                except Exception as e:
                    print(f"[Pipeline]\t [Warning]\t Heatmap generation failed: {e}")
            else:
                print(f"[Pipeline]\t [Warning]\t Cannot read image from {raw_image_path}, return heatmap 'None'")

        fold_embeddings = None
        if self.extract_embeddings:
            fold_embeddings = self._extract_fold_embeddings(img_tensor, patient_data, used_mode)

        return {
            "is_abnormal": is_abnormal,
            "abnormal_probability": round(abnormal_prob, 4),
            "probability": round(abnormal_prob, 4) if is_abnormal else round(1 - abnormal_prob, 4),
            "inference_mode": used_mode,
            "suggestion": self.abnormal_suggestion if is_abnormal else self.normal_suggestion,
            "heatmap": heatmap,
            "fold_embeddings": fold_embeddings,
        }

    def _tabular_for_fold(self, fold_idx: int, patient_data: dict) -> torch.Tensor:
        processor_idx = fold_idx if fold_idx < len(self.tabular_processors) else 0
        return self.preprocess_tabular(self.tabular_processors[processor_idx], patient_data, self.device)

    def _extract_fold_embeddings(self, img_tensor: torch.Tensor, patient_data: dict, used_mode: str) -> list[dict]:
        '''
        Extract each fold model's own pre-classifier embedding and its own calibrated
        prediction (bypassing ensemble averaging), so per-fold t-SNE panels reflect that
        fold's model rather than a single representative fold.

        Returns:
            list[dict]: one {"embedding": np.ndarray, "pred_label": int} per fold, in fold order
        '''
        if used_mode == "multimodal":
            ensemble = self.multimodal_ensemble
            calib = self.mm_calib
        else:
            ensemble = self.image_only_ensemble
            calib = self.im_calib

        temperatures = calib.temperatures
        threshold = calib.threshold if calib.threshold is not None else 0.5

        results = []
        with torch.inference_mode():
            for i, model in enumerate(ensemble.models):
                if used_mode == "multimodal":
                    tab_tensor = self._tabular_for_fold(i, patient_data)
                    embedding = model.get_embedding(img_tensor, tab_tensor)
                    logits = model(img_tensor, tab_tensor)
                else:
                    embedding = model.get_embedding(img_tensor)
                    logits = model(img_tensor, None)

                t = temperatures[i] if temperatures is not None and i < len(temperatures) else 1.0
                probs = torch.softmax(logits / t, dim=1)
                abnormal_prob = probs[0, self.abnormal_idx].item()
                pred_label = int(abnormal_prob >= threshold)

                results.append({
                    "embedding": embedding.cpu().numpy()[0],
                    "pred_label": pred_label,
                })

        return results

    def _forward(self, img_tensor: torch.Tensor, patient_data: dict) -> tuple[torch.Tensor, str, float]:

        with torch.inference_mode():
            if self.multimodal_ensemble is not None and self._has_valid_tabular(patient_data) and not self.image_only:

                tab_tensors = []
                for processor in self.tabular_processors:
                    tab_tensors.append(self.preprocess_tabular(processor, patient_data, self.device))
                
                temps = list(self.mm_calib.temperatures) if self.mm_calib.temperatures is not None else None
                thresh = self.mm_calib.threshold

                probs = self.multimodal_ensemble(img_tensor, tab_tensors, temps)
                used_mode = "multimodal"
            elif self.image_only_ensemble is not None:
                temps = list(self.im_calib.temperatures) if self.im_calib.temperatures is not None else None
                thresh = self.im_calib.threshold

                probs = self.image_only_ensemble(img_tensor, None, temps)
                used_mode = "image_only"
            else:
                raise ValueError(f"[Pipeline]\t [Error]\t Need at least one ensemble (multimodal or image_only) but got both None")
        
        return probs, used_mode, thresh

    def _evaluate(self, probs: torch.Tensor, threshold: float) -> tuple[float, bool]:

        probs = probs.cpu().numpy()[0]

        abnormal_prob = float(probs[self.abnormal_idx])
        is_abnormal = bool(abnormal_prob >= threshold) if threshold is not None else bool(abnormal_prob >= 0.5)

        return abnormal_prob, is_abnormal

    def _has_valid_tabular(self, patient_data: dict) -> bool:
        for k in self.required_keys:
            if k not in patient_data or patient_data[k] is None:
                return False
        
        return True

    def _resolve_device(self, current_device: str):
        """
        Detect available hardware and update 'device' field if it is set to 'auto'.
        Priority: CUDA > MPS > CPU
        """
        resolved = current_device

        if current_device == "auto":
            if torch.cuda.is_available():
                resolved = "cuda"
                print(f"[Pipeline] Device 'auto' resolved to: CUDA ({torch.cuda.get_device_name(0)})")
            elif torch.backends.mps.is_available():
                resolved = "mps"
                print("[Pipeline] Device 'auto' resolved to: MPS (Apple Silicon)")
            else:
                resolved = "cpu"
                print("[Pipeline] Device 'auto' resolved to: CPU")
            
        return resolved
    
