import numpy as np
import sys
import cv2
from pathlib import Path


cur_file_path = Path(__file__).resolve()
project_root  = cur_file_path.parent.parent.parent
tkr_seg_path  = project_root / "external" / "TKR_Segmentation"

if str(tkr_seg_path) not in sys.path:
    sys.path.insert(0, str(tkr_seg_path))


from tkr_inference import TKRSegmentationPipeline
from .base_cropper import BaseWoundCropper

class SegmentationWoundCropper(BaseWoundCropper):
    def __init__(self, ckpt_path: str | None = None, padding: int = 40, device: str = "auto"):
        self.segmenter = TKRSegmentationPipeline(checkpoint=ckpt_path, device=device)
        self.padding = padding

    def crop_single_image(self, img_path: str, **kwargs) -> np.ndarray | None:
        try:
            img = cv2.imread(img_path)
            result = self.segmenter.predict(img)
        except ValueError:
            print(f"[Seg Cropper]\t [Warning]\t segmenter predict failed at {img_path}")
            return None

        if result.decision != "ACCEPT_TKR_WOUND_FOUND" or result.bbox_xyxy is None:
            print(f"[Seg Cropper]\t [Warning]\t segmenter cannot find wound at {img_path}")
            return None

        return result.masked_bgr