# TKR Wound Care System

全膝關節置換（Total Knee Replacement, TKR）術後傷口照護系統。涵蓋傷口照片／病患資料的前處理、影像＋表格多模態的風險判讀模型訓練與分析，以及結合傷口分割（透過 `external/TKR_Segmentation` submodule）與風險判讀的推論 API。

## 目錄結構

```
core/                  訓練核心（trainer、runner、models、loss、callbacks 等）
inference/             推論 pipeline 與 FastAPI 服務（main.py）
analysis/              訓練結果分析（cross-validation 彙整、分類報告、錯誤檢視）
data_preprocessing/    原始資料轉換、切分、資料分布檢視（見該資料夾內 README.md）
config/                訓練/推論用的實驗設定檔
external/TKR_Segmentation/   傷口分割／OOD 驗證的獨立套件（git submodule）
tkr_train_sh/          實驗批次執行腳本
TKR_train.py           訓練入口
requirements.txt       Python 依賴
```

## 環境需求

- Python 3.10+（`external/TKR_Segmentation` 要求 ≥3.10）
- pip
- 訓練／推論建議搭配 GPU；在 macOS 上可用 `PYTORCH_ENABLE_MPS_FALLBACK=1` 搭配 MPS

## 安裝與下載（Clone & Submodule）

```bash
git clone git@github.com:Mason1217/tkr-wound-care-system.git
cd tkr-wound-care-system
git submodule update --init --recursive
```

也可以用 `git clone --recurse-submodules git@github.com:Mason1217/tkr-wound-care-system.git` 一步完成。

日後 `git pull` 更新主專案後，submodule 不會自動跟著更新，需再跑一次：

```bash
git submodule update --init --recursive
```

安裝依賴：

```bash
pip install -r requirements.txt
pip install -e external/TKR_Segmentation   # 提供 tkr_inference 套件，run_inference.py / gather_models.py 等會用到
```

## 環境變數

複製 `.env.example` 為 `.env`，並視需要調整內容（例如 `PUBLIC_BASE_URL`，見下方 API 章節）。

```bash
cp .env.example .env
```

## 使用方式

### 資料前處理

0. 準備資料，目錄結構如下：
```
data
└── TKR
    ├── 07
    ├── 08
    ├── 09
    ├── 10
    ├── 11
    ├── 工作表1 (雲端在外面的照片)
    └── excel檔案 (TKR傷口分析0710.xlsx，有兩個，注意是十月新增的)
```

1. 執行以下指令：
```bash
chmod +x tkr_build.sh

./tkr_build.sh
```

### 模型訓練（TKR_train.py）

```bash
python TKR_train.py --base config/tkr_base_config.yaml --exp "path to experiment config file"

ex:
python TKR_train.py --base config/tkr_base_config.yaml --exp config/exp015_ablation/seed40/training/loss_test_ce.yaml

ex: (for mac)
caffeinate python TKR_train.py --base config/tkr_base_config.yaml --exp config/exp015_ablation/seed40/training/loss_test_ce.yaml
```

### 模型分析（analysis.cv_analysis）

```bash
python -m analysis.cv_analysis \
    --base config/tkr_base_config.yaml \
    --exp config/exp015_ablation/seed40/training/loss_test_ce.yaml \
    --report checkpoints/stage2_finetune/analysis_results/threshold_0.5/classification_report_image_level.json
```

完整批次訓練＋分析流程範例可參考 `tkr_train_sh/exp015_ablation.sh`。

### Local prediction API

1. Copy `.env.example` to `.env` and set `PUBLIC_BASE_URL` to the HTTPS ngrok URL.
2. Run the API:
```bash
uvicorn inference.main:app --host 0.0.0.0 --port 8000
```
3. Start ngrok:
```bash
ngrok http 8000
```

`POST /predict` accepts `multipart/form-data` fields:
```
image
days_post_op
wound_size
bmi
```

The response contains:
```
diagnosis
confidence
resultImageUrl
error
```

## External Submodule: TKR_Segmentation

`external/TKR_Segmentation`（[Mason1217/TKR_Segmentation](https://github.com/217Timothy/TKR_Segmentation)）是獨立的傷口分割／OOD（out-of-distribution）驗證套件：輸入 TKR 術後傷口照片，先判斷是否為有效的傷口照片，再輸出分割遮罩、疊圖與去背裁切結果。詳細安裝與 CLI 用法請見該 submodule 自己的 README。
