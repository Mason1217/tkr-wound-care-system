import os
import shutil
import traceback
import uuid
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


load_dotenv()

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
RESULT_DIR = Path(os.getenv("RESULT_DIR", "static/results"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", "static"))
INFERENCE_CONFIG = os.getenv("INFERENCE_CONFIG")
ENABLE_HEATMAP = os.getenv("ENABLE_HEATMAP", "true").lower() in {"1", "true", "yes", "on"}

if not INFERENCE_CONFIG:
    asset_config = Path("inference_asset/inference_config.yaml")
    INFERENCE_CONFIG = str(asset_config if asset_config.exists() else Path("config/inference_config.yaml"))

RESULT_DIR.mkdir(parents=True, exist_ok=True)

from inference.run_inference import inference_pipeline_setup

app = FastAPI(title="TKR Wound Analysis API")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

try:
    inference_pipeline = inference_pipeline_setup(INFERENCE_CONFIG, heatmap=ENABLE_HEATMAP)
    print("[API] TKR inference pipeline initialized.")
except Exception as exc:
    print(f"[API] Failed to initialize TKR inference pipeline: {exc}")
    raise


def _build_result_url(filename: str) -> str:
    if not PUBLIC_BASE_URL:
        raise RuntimeError("PUBLIC_BASE_URL is not configured")
    return f"{PUBLIC_BASE_URL}/static/results/{filename}"


def _save_result_image(result: dict, result_path: Path) -> bool:
    heatmap = result.get("heatmap")
    if heatmap is None:
        return False
    if not isinstance(heatmap, np.ndarray) or heatmap.ndim != 3 or heatmap.shape[2] != 3:
        print(f"[API] Invalid heatmap shape: {getattr(heatmap, 'shape', None)}")
        return False

    # The pipeline returns RGB heatmaps; cv2.imwrite expects BGR.
    heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    return bool(cv2.imwrite(str(result_path), heatmap_bgr))


def _format_prediction_response(result: dict, result_image_url: str | None, error: str | None) -> dict:
    is_abnormal = bool(result.get("is_abnormal", False))
    abnormal_prob = float(result.get("abnormal_probability", 0.0))

    diagnosis = 1 if is_abnormal else 0
    confidence = abnormal_prob if is_abnormal else 1 - abnormal_prob

    return {
        "diagnosis": diagnosis,
        "confidence": round(float(confidence), 4),
        "resultImageUrl": result_image_url,
        "error": error,
    }


def _error_response(error: str) -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={
            "diagnosis": -1,
            "confidence": 0,
            "resultImageUrl": None,
            "error": error,
        },
    )


def _log_prediction(
    original_shape,
    heatmap_shape,
    diagnosis: int,
    confidence: float,
    result_image_url: str | None,
    error: str | None,
):
    print(
        "[API] Prediction result "
        f"original_shape={original_shape} "
        f"heatmap_shape={heatmap_shape} "
        f"diagnosis={diagnosis} "
        f"confidence={confidence} "
        f"resultImageUrl={result_image_url} "
        f"error={error}"
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    days_post_op: int = Form(...),
    wound_size: float = Form(...),
    bmi: float = Form(...),
):
    request_id = uuid.uuid4().hex
    input_path = RESULT_DIR / f"input_{request_id}.jpg"
    result_filename = f"result_{request_id}.jpg"
    result_path = RESULT_DIR / result_filename

    try:
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        original_img = cv2.imread(str(input_path))
        original_shape = getattr(original_img, "shape", None)
        if original_img is None:
            _log_prediction(original_shape, None, -1, 0, None, "IMAGE_LOAD_FAILED")
            return _error_response("IMAGE_LOAD_FAILED")

        patient_data = {
            "術後天數": int(days_post_op),
            "傷口大小": float(wound_size),
            "BMI": float(bmi),
        }

        result = inference_pipeline.predict(str(input_path), patient_data)
        if not isinstance(result, dict):
            raise ValueError(f"Unknown model output type: {type(result)}")

        result_image_url = None
        error = None
        if _save_result_image(result, result_path):
            result_image_url = _build_result_url(result_filename)
        else:
            error = "RESULT_IMAGE_NOT_GENERATED"

        response = _format_prediction_response(result, result_image_url, error)
        _log_prediction(
            original_shape,
            getattr(result.get("heatmap"), "shape", None),
            response["diagnosis"],
            response["confidence"],
            response["resultImageUrl"],
            response["error"],
        )
        return response

    except ValueError as exc:
        if "WOUND_NOT_FOUND" in str(exc):
            _log_prediction(None, None, -1, 0, None, "WOUND_NOT_FOUND")
            return _error_response("WOUND_NOT_FOUND")
        print(f"[API] Image processing failed: {exc}")
        _log_prediction(None, None, -1, 0, None, "IMAGE_PROCESSING_FAILED")
        return _error_response("IMAGE_PROCESSING_FAILED")

    except RuntimeError as exc:
        print(f"[API] Runtime error: {exc}")
        _log_prediction(None, None, -1, 0, None, "PREDICTION_FAILED")
        return _error_response("PREDICTION_FAILED")

    except Exception as exc:
        print("[API] Prediction failed.")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    finally:
        if input_path.exists():
            input_path.unlink()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
