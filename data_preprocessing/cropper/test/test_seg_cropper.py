import cv2
from pathlib import Path
import sys

# Setup project root in sys.path if necessary, or rely on PYTHONPATH
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

# Import the class as an external consumer would
from data_preprocessing.cropper import SegmentationWoundCropper

def main():
    """Execute cropper testing pipeline."""
    img_path = str(project_root / "data" / "TKR" / "processed" / "images" / "工作表1_IMG_4231.jpg")
    output_path = Path(__file__).resolve().parent / "test_out.jpg"
    
    cropper = SegmentationWoundCropper(ckpt_path=None)
    cropped_img = cropper.crop_single_image(img_path)
    
    if cropped_img is None:
        print(f"[Seg Cropper Test]\t crop_single_image returned None for {img_path}")
        return
        
    print(f"[Seg Cropper Test]\t crop shape={cropped_img.shape}  dtype={cropped_img.dtype}")
    cv2.imwrite(str(output_path), cropped_img)
    print(f"[Seg Cropper Test]\t saved -> {output_path}")

if __name__ == "__main__":
    main()