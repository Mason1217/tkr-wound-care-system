from PIL.Image import Image
import torchvision.transforms as transforms
from omegaconf import DictConfig
import torchvision.transforms.functional as F

class SquarePad:
    def __init__(self, padding_mode):
        self.padding_mode = padding_mode
    
    def __call__(self, image: Image):
        w, h = image.size
        max_wh = max(w, h)
        p_left, p_top = (max_wh - w) // 2, (max_wh - h) // 2
        p_right, p_bottom = max_wh - w - p_left, max_wh - h - p_top
        return F.pad(image, [p_left, p_top, p_right, p_bottom], padding_mode=self.padding_mode)

def build_transforms(cfg: DictConfig, is_train: bool = True):
    """
    Args:
        cfg: data config (includes settings like image_size..., etc)
        is_train: true for augmentation

    **Note:** image type should be **PIL.Image**
    """
    img_size = tuple(cfg.image_size)
    transform_list = []

    # Padding & Resize
    transform_list.append(SquarePad(cfg.get("padding_mode", "constant")))
    transform_list.append(transforms.Resize(img_size))

    if is_train and cfg.get("basic_transforms", True):
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_list.append(transforms.RandomRotation(degrees=15))
        transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
        
    elif is_train and not cfg.get("basic_transforms", True):
        print("[Image Transform]\t [Warning]\t No basic data augmentations.")

    # To Tensor
    transform_list.append(transforms.ToTensor())
    
    if is_train:
        use_erasing = cfg.get("image_transforms", {}).get("random_erasing", False)
        if use_erasing:
            print("[Image Transform]\t Applying Random Erasing in training transforms.")
            transform_list.append(transforms.RandomErasing(
                p=0.3, 
                scale=(0.02, 0.15), 
                ratio=(0.3, 3.3), 
                value="random"
            ))

    # Normalize
    transform_list.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))

    return transforms.Compose(transform_list)