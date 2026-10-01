import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # [Fix] MPS not support bicubic interpolation (when training dino)

import glob
import torch
import argparse
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf

from core import ConfigManager
from core.data.processor import TKRDataProcessor
from core.data.transforms import build_transforms
from core.models import get_model
from data_preprocessing.cropper.base_cropper import BaseWoundCropper
from data_preprocessing.cropper.seg_cropper import SegmentationWoundCropper

from .gather_models import find_all_models
from .detector import BaseWoundDetector, YOLOWoundDetector
from .pipeline.calibrate import optimize_temperature, find_best_threshold
from .pipeline.ensemble import TKRFoldEnsemble
from .pipeline.pipeline import TKRInferencePipeline
from .evaluate.system_evaluator import TKRSystemEvaluator
from .evaluate.system_evaluator import HEIGHT_COL, WEIGHT_COL, BMI_COL, POST_OP_COL, WOUND_SZ_COL, SEXUAL_COL, LABEL_COL, LABEL_MAPPING
from .evaluate.heatmap import get_heatmap_visualizer, BaseHeatmapVisualizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config", type=str, required=True, help="path to inference config file")
    parser.add_argument("--update_calib", action="store_true", help="whether to update calibrate parameters (temperature & threshold)")
    parser.add_argument("--image_only", action="store_true", help="use image_only ensemble instead of multimodal")
    parser.add_argument("--heatmap", action="store_true", help="whether to visualize heatmap or not")
    parser.add_argument("--tsne", action="store_true", help="whether to run t-SNE embedding analysis or not")
    parser.add_argument("--beta", type=float)

    return parser.parse_args()

def extract_oof_logits(
        model: torch.nn.Module,
        processor: TKRDataProcessor,
        detector: BaseWoundDetector | None,
        cropper: BaseWoundCropper | None,
        image_transforms: transforms.Compose,
        csv_path: str,
        device: torch.device,

) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Extract prediction logits & labels in validation set
    '''

    df = pd.read_csv(csv_path)
    all_logits, all_labels = [], []

    model.eval()
    for _, row in df.iterrows():

        image_path = row["image_path"]
        patient_data = {
            HEIGHT_COL   : row.get(HEIGHT_COL, None),
            WEIGHT_COL   : row.get(WEIGHT_COL, None),
            BMI_COL      : row.get(BMI_COL, None),
            POST_OP_COL  : row.get(POST_OP_COL, None),
            WOUND_SZ_COL : row.get(WOUND_SZ_COL, None),
            SEXUAL_COL   : row.get(SEXUAL_COL, None),
        }
        label = row.get(LABEL_COL)
        if label is None:
            print(f"[Run Inference]\t [Warning]\t Cannot get label in {row}")
            continue

        true_label = LABEL_MAPPING.get(label)

        try:
            img_tensor = TKRInferencePipeline.preprocess_image(detector, cropper, image_transforms, image_path, device)
            tab_tensor = TKRInferencePipeline.preprocess_tabular(processor, patient_data, device)

            with torch.inference_mode():
                logits = model(img_tensor, tab_tensor)
                all_logits.append(logits.cpu())
                all_labels.append(true_label)
        
        except ValueError as e:
            if str(e) == "WOUND_NOT_FOUND":
                print(f"[Run Inference]\t [Warning]\t Cannot detect wound in {image_path}")
                continue
            elif str(e) == "IMAGE_LOAD_FAILED":
                print(f"[Run Inference]\t [Warning]\t Cannot load image at {image_path}")
                continue
            raise e
        
        except Exception as e:
            print(f"[Run Inference]\t [Error]\t {e}")
            raise e
    
    return torch.cat(all_logits, dim=0), torch.tensor(all_labels, dtype=torch.long)

def auto_calibrate_ensemble(
        cfg: DictConfig,
        ensemble_model: TKRFoldEnsemble,
        detector: BaseWoundDetector | None,
        cropper: BaseWoundCropper | None,
        tabular_processors: list[TKRDataProcessor],
        image_transforms: transforms.Compose,
        device: torch.device | str,
        target_recall: float = 0.8,
) -> dict:
        
    if isinstance(device, str): device = torch.device(device)
    ensemble_model = ensemble_model.to(device)

    val_csv_paths   = cfg.val_csv_path
    temperatures    = []
    all_calib_probs = []
    all_labels      = []

    for i, model in enumerate(ensemble_model.models):
        print(f"[Run Inference]\t Fold {i} Calibrating...")
        logits_tensor, labels_tensor = extract_oof_logits(model, tabular_processors[i], detector, cropper, image_transforms, val_csv_paths[i], device)
        
        t_i = optimize_temperature(logits_tensor, labels_tensor)
        temperatures.append(t_i)

        calib_logits = logits_tensor / t_i
        calib_probs  = torch.softmax(calib_logits, dim=1).numpy()
        all_calib_probs.append(calib_probs)
        all_labels.append(labels_tensor.numpy())
    
    combined_probs  = np.concatenate(all_calib_probs, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    best_threshold, _ = find_best_threshold(combined_probs, combined_labels, beta=2.0)

    calib_params = {
        "temperatures": [round(t, 4) for t in temperatures],
        "threshold"   : round(best_threshold, 4),
    }

    print(f"[Run Inference]\t Auto calibrated: best threshold: {best_threshold:.3f} | temperatures: {temperatures}")
    return calib_params

def load_ensemble(model_cfg: DictConfig | None, model_dir: str | None) -> TKRFoldEnsemble | None:

    if model_dir is None or model_cfg is None:
        print("[Run Inference]\t [Warning]\t either model_cfg or model_dir is None while loading ensemble.")
        return None

    weight_files = glob.glob(os.path.join(model_dir, "*.pth"))

    if len(weight_files) == 0: raise FileNotFoundError(f"No model (.pth) found in {model_dir}")
    print(f"[Run Inference]\t Found {len(weight_files)} fold weight(s) in {model_dir}")

    models = []
    print(f"[Run Inference]\t [Load Ensemble]\t found weight files:\n")

    for weight_file in sorted(weight_files):
        print(f"[Run Inference]\t [Load Ensemble]\t {weight_file}")
        try:
            checkpoint = torch.load(weight_file, map_location="cpu")
            
            model = get_model(model_cfg)
            model.load_state_dict(checkpoint["model_state_dict"])

            models.append(model)
        
        except Exception as e:
            print(f"[Run Inference]\t [Error]\t {e}")
    
    model_ensemble = TKRFoldEnsemble(models)
    return model_ensemble

def load_tab_processors(processor_dir: str) -> list[TKRDataProcessor]:

    pkl_files = glob.glob(os.path.join(processor_dir, "*.pkl"))
    processors = []
    print(f"[Run Inference]\t [Load Tabular Processor]\t found pickle files:\n")

    for pkl_file in sorted(pkl_files):
        print(f"[Run Inference]\t [Load Tabular Processor]\t {pkl_file}")

        processor = TKRDataProcessor.load(pkl_file)
        processors.append(processor)
    
    return processors

def load_calib_params(
        cfg: DictConfig,
        cfg_manager: ConfigManager,
        inference_config_path: str,
        multimodal_ensemble: TKRFoldEnsemble | None,
        image_only_ensemble: TKRFoldEnsemble | None,
        detector: BaseWoundDetector | None,
        cropper: BaseWoundCropper | None,
        tabular_processors: list[TKRDataProcessor],
        image_transforms: transforms.Compose,
        device: str,
        update: bool = False,

) -> DictConfig:

    calib_params = cfg.get("calibrate_parameters")

    if not update:
        print("[Run Inference]\t No updates in calibrate parameter.")
        return calib_params

    mm_params = im_params = {"temperatures": None, "threshold": None}

    if multimodal_ensemble is not None:
        print("[Run Inference]\t Calibrating multi modal ensemble")
        mm_params = auto_calibrate_ensemble(cfg, multimodal_ensemble, detector, cropper, tabular_processors, image_transforms, device)
    if image_only_ensemble is not None:
        print("[Run Inference]\t Calibrating image only ensemble")
        im_params = auto_calibrate_ensemble(cfg, image_only_ensemble, detector, cropper, tabular_processors, image_transforms, device)

    calib_params.multimodal = mm_params
    calib_params.image_only = im_params

    OmegaConf.update(cfg, "calibrate_parameters", calib_params)
    cfg_manager.save(inference_config_path)

    return calib_params

def inference_pipeline_setup(inference_config_path: str, update_calib_params: bool = False, image_only: bool = False, heatmap: bool = False, tsne: bool = False) -> TKRInferencePipeline:

    cfg_manager = ConfigManager(inference_config_path)
    cfg = cfg_manager.cfg

    device                  = cfg.device
    multimodal_model_dir    = cfg.multimodal_model_dir
    multimodal_model_cfg    = cfg.multimodal_model
    image_only_model_dir    = cfg.image_only_model_dir
    image_only_model_cfg    = cfg.image_only_model
    tabular_processor_dir   = cfg.tabular_processor_dir
    image_transform_cfg     = cfg.image_transforms

    multimodal_ensemble     = load_ensemble(multimodal_model_cfg, multimodal_model_dir)
    image_only_ensemble     = load_ensemble(image_only_model_cfg, image_only_model_dir)
    detector                = YOLOWoundDetector(cfg.yolo_model_path) if cfg.yolo_model_path else None
    cropper                 = SegmentationWoundCropper()
    tabular_processors      = load_tab_processors(tabular_processor_dir)
    image_transforms        = build_transforms(image_transform_cfg, is_train=False)
    calib_params            = load_calib_params(cfg, cfg_manager, inference_config_path, multimodal_ensemble, image_only_ensemble, detector, cropper, tabular_processors, image_transforms, device, update_calib_params)

    pipeline = TKRInferencePipeline(
        multimodal_ensemble=multimodal_ensemble,
        image_only_ensemble=image_only_ensemble,
        detector=detector,
        cropper=cropper,
        tabular_processors=tabular_processors,
        image_transforms=image_transforms,
        calib_params=calib_params,
        device=device,
        image_only=image_only,
        extract_embeddings=tsne,
    )

    if heatmap:
        pipeline.heatmap_viz = load_heatmap_viz(cfg, pipeline.device)

    return pipeline

def load_heatmap_viz(cfg: DictConfig, device: torch.device) -> BaseHeatmapVisualizer:
    model_cfg = cfg.analyze_model
    checkpoint = torch.load(model_cfg.path, map_location="cpu")

    model = get_model(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval() # close dropout

    target_layer = model_cfg.get("target_layer", None)

    return get_heatmap_visualizer(model_cfg.name, model, target_layer)

def main():
    args = parse_args()
    cfg_path = args.inference_config
    cfg = ConfigManager(cfg_path)

    test_csv_paths  = cfg.test_csv_path
    save_dirs       = cfg.result_dir

    pipeline = inference_pipeline_setup(cfg_path, args.update_calib, args.image_only, args.heatmap, args.tsne)

    for test_csv_path, save_dir in zip(test_csv_paths, save_dirs):
        os.makedirs(save_dir, exist_ok=True)

        evaluator = TKRSystemEvaluator(pipeline, test_csv_path)
        evaluator.evaluate(save_dir, cfg.reliability_diagram_filename, cfg.auc_pr_filename, cfg.auc_roc_filename, cfg.get("tsne_filename"), cfg.get("tsne_perplexities"))
        evaluator.profile_model()

if __name__ == "__main__":
    main()