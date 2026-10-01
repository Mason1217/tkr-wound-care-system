## Commands

### data preprocessing

0. Prepare data like below
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

1. Run following commands
```
chmod +x tkr_build.sh

./tkr_build.sh
```

### tkr_train.py

```
python tkr_train.py --exp "path to experiment config file"

ex:
python tkr_train.py --exp config/experiments/exp002/config.yaml

ex: (for mac)
caffeinate python tkr_train.py --exp config/experiments/exp002/config.yaml

```

### tkr_analysis.py

Run following commands
```
chmod +x tkr_analysis_sh/tkr_analysis_exp002.sh

./tkr_analysis_sh/tkr_analysis_exp002.sh
```

### docker commands

* 建立docker
```
docker build -t wound_detector .
```

* 開啟container
```
docker run --rm -it \
    --gpus all \
    -v ~/project/Wound_Detection/data:/app/data \
    -v ~/project/Wound_Detection/results:/app/results \
    wound_detector /bin/bash 
    # 或者執行您的 run.sh 腳本:
    # wound_detector ./run.sh
```

### Local prediction API

1. Copy `.env.example` to `.env` and set `PUBLIC_BASE_URL` to the HTTPS ngrok URL.
2. Run the API:
```
uvicorn inference.main:app --host 0.0.0.0 --port 8000
```
3. Start ngrok:
```
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
