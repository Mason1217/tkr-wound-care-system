from ultralytics import YOLO
import os

def main():
    # yolo11n.pt (Nano) 最快，適合只有 CPU 或輕量需求
    # yolo11s.pt (Small) 稍大，準確度較高
    model = YOLO("yolo11s.pt") 

    yaml_path = os.path.abspath("yolo/wound_data.yaml")

    print(f"開始訓練，使用設定檔: {yaml_path}")
    results = model.train(
        data=yaml_path,
        epochs=100,         # 訓練輪數，通常 50-100 足夠
        imgsz=640,          # 圖片輸入大小
        batch=16,           # Batch size
        device="cpu",       # 如果有 GPU 改為 0 或 'cuda'
        project="YOLO",     # 訓練結果儲存的專案資料夾
        name="wound_det",   # 實驗名稱
        exist_ok=True,      # 是否覆蓋同名實驗
        patience=25,        # Early stopping
        save_period=1,
    )
    
    print("訓練完成！")
    print(f"最佳模型已儲存於: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    main()