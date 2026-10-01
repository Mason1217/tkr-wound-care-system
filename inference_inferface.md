# TKR Model Analysis Module

### Version: 1.1

### Assignees:
1. **Mason** (ML Model Development)
2. **An** (App Development)

## 0. Setup & Initialization
* requirements.txt
---
* Prepare directory `inferenct_asset`
**Note:** 可查看雲端硬碟中 `TKR Inference Asset` 資料夾
**Note:** 目前 image_only 還不太能用，先放著讓程式不要報錯
**Note:** inference 設定檔的部分可以參考 `inference_config.yaml`
```
inference_asset
├── models
│   ├── image_only
│   │   └── unknown_run_stage2_finetune_best_model.pth
│   └── multimodal
│       ├── run1_split_versionseed_40_fold_0_stage2_finetune_best_model.pth
│       ├── run2_split_versionseed_40_fold_1_stage2_finetune_best_model.pth
│       ├── run3_split_versionseed_40_fold_2_stage2_finetune_best_model.pth
│       ├── run4_split_versionseed_40_fold_3_stage2_finetune_best_model.pth
│       └── run5_split_versionseed_40_fold_4_stage2_finetune_best_model.pth
├── processors
│   ├── run1_split_versionseed_40_fold_0_tabular_processor.pkl
│   ├── run2_split_versionseed_40_fold_1_tabular_processor.pkl
│   ├── run3_split_versionseed_40_fold_2_tabular_processor.pkl
│   ├── run4_split_versionseed_40_fold_3_tabular_processor.pkl
│   └── run5_split_versionseed_40_fold_4_tabular_processor.pkl
└── yolo
    └── v1.pt
```

## 1. Module Architecture
* 需要使用目前 Wound_Detection 內的程式碼
* 主要使用其中 tkr_inference 這個模組的功能

```
tkr_inference/
├── __init__.py
├── ensemble.py
├── gather_models.py
├── pipeline.py
├── run_inference.py
└── system_evaluator.py
```

**Note:** 可參考 `run_inference` 的寫法，主要需使用 pipeline 中 `TKRInferencePipeline` 這個類別

## 3. Function Call
```python
import os

from tkr_inference.run_inference import inference_pipeline_setup
from tkr_inference.pipeline import TKRInferencePipeline

inference_config_path = os.path.join("inference_asset", "inference_config")
inference_pipeline = inference_pipeline_setup(inference_config_path)

inference_pipeline.predict(image_path, patient_data)
```

## 4. Error & Exception Handling
這部分可參考 `system_evaluator.py` 中的 `SystemEvaluator` 處理方法

* ValueError("WOUND_NOT_FOUND"): 當 yolo 沒有偵測到傷口時會拋出。