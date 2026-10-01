from abc import ABC, abstractmethod

class BaseWoundDetector(ABC):
    @abstractmethod
    def has_wound(self, img_path: str, **kwargs) -> bool:
        '''
        Check if there exists wound in the image

        Returns:
            bool: **True** means there exists wound and may proceed next step<br>
                  **False** means it's not wound image, should be intercepted and stop process
        '''
        pass