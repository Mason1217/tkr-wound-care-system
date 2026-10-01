## `data_engineering` module
---
### `build.sh`
Processing raw data, including:
1. Convert images from .heic to .jpg
2. Check images' validity & establish id-path map for later usage
3. Check metadata's validity

**How to run command**
```bash
# Run build.sh
cd Wound_Detection
chmod +x data_preprocessing/build.sh
./data_preprocessing/build.sh
```

Path / filename you can modify
```bash
# You can modify the following arguments:
RAW_DATA_ROOT="directory contains all raw data"
PROCESSED_DIR="directory holds all processed data"

RAW_METADATA="filename of raw metadata (e.g. TKR傷口分析0325.xlsx)"
MASTER_METADATA_FILE="filename of processed metadata (e.g. master_metadata.csv)"

RAW_IMG_DIRS="directories contain raw images (collected by nurse and separated with month)"
IMG_MAP_FILE="filename of id-path json file (e.g. image_id_map.json)"
PROCESSED_IMG_DIR="directory holds processed images"

SPLIT_VERSION="split version (e.g. original_v0 or segmentation_v1...)"
```

### `constant.py`
Contains *hyperparameters* or *constants* for data preprocessing

1. Hyperparameters for train-test split
```python
# You can modify the following
SPLIT_RANDOM_ST = # random seed for split
VAL_SET_SIZE    = # size of validation set
TEST_SET_SIZE   = # size of testing set
STRATIFY_COL    = “建議”
```

### `show_data_distribution.py`
Generate distribution plots for 8 key variables in `master_metadata.csv`:
- `overview_distribution.png` — combined 2x2 grid: risk-group counts (`建議`), `BMI`, `術後天數`, `傷口大小`
- `crack_distribution.png` — `裂開`
- `exudate_distribution.png` — `滲液`
- `swelling_severity_distribution.png` — `紅腫程度`
- `swelling_extent_distribution.png` — `紅腫範圍`

Every bin/bar in all 5 images (including the BMI, 術後天數, and 傷口大小 histograms) stacks a green (Normal) segment below a red (Abnormal) segment, sized by how many samples in that bin are risk-normal vs. risk-abnormal (per `建議`); the categorical charts additionally annotate each bar's total count on top. All chart text (titles, axis labels, legends, category names) is rendered in English regardless of the underlying Chinese column/category names. All images are saved next to the input metadata file.

**How to run command**
```bash
python -m data_preprocessing.data_engineering.show_data_distribution --metadata "path_to_master_metadata.csv"
```