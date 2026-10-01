from abc import ABC, abstractmethod
import numpy as np

class BaseWoundCropper(ABC):
    @abstractmethod
    def crop_single_image(self, img_path: str, **kwargs) -> np.ndarray | None:
        pass