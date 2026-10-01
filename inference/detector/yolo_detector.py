import cv2
from ultralytics import YOLO
from pathlib import Path

from .base_detector import BaseWoundDetector

class YOLOWoundDetector(BaseWoundDetector):
    def __init__(self, model_path: str = None, conf_threshold: float = 0.25):

        if model_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            self.model_path = str(project_root / "data_preprocessing" / "yolo" / "models" / "v1.pt")
        else:
            self.model_path = model_path

        self.model = YOLO(self.model_path)
        self.conf_threshold = conf_threshold

    def has_wound(self, img_path: str, **kwargs) -> bool:
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WoundDetector]\t [Error]\t while loading {img_path}...")
            return False
        
        try:
            results = self.model.predict(img, conf=self.conf_threshold, verbose=False)

            if len(results) > 0 and len(results[0].boxes) > 0:
                return True
            return False
        
        except Exception as e:
            print(f"[WoundDetector]\t [Error]\t {e}")
            return False