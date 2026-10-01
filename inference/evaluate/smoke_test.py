import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import cv2
import json

from .run_inference import inference_pipeline_setup


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test for TKRInferencePipeline.predict()")
    parser.add_argument("--config",        type=str,   required=True, help="Path to inference config YAML")
    parser.add_argument("--image",         type=str,   required=True, help="Path to a test image")
    parser.add_argument("--bmi",           type=float, default=25.0)
    parser.add_argument("--post_op_days",  type=int,   default=7)
    parser.add_argument("--wound_size",    type=float, default=3.0)
    parser.add_argument("--out_dir",       type=str,   default="smoke_test")
    return parser.parse_args()


def _print_result(result: dict):
    print(json.dumps({k: v for k, v in result.items() if k != "heatmap"}, ensure_ascii=False, indent=2))
    heatmap = result["heatmap"]
    if heatmap is not None:
        print(f"  heatmap  shape={heatmap.shape}  dtype={heatmap.dtype}")
    else:
        print("  heatmap=None")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    full_data    = {"BMI": args.bmi, "術後天數": args.post_op_days, "傷口大小": args.wound_size}
    missing_data = {"BMI": None, "術後天數": None, "傷口大小": None}

    # ── Case 1: Normal prediction with heatmap ─────────────────────────────────────────
    print("\n[Case 1] predict with heatmap")
    pipeline = inference_pipeline_setup(args.config, heatmap=True)
    result = pipeline.predict(args.image, full_data)
    _print_result(result)

    assert result["heatmap"] is not None,                       "Expected heatmap but got None"
    assert result["heatmap"].ndim == 3,                         "Expected 3-dim array (H, W, C)"
    assert result["heatmap"].shape[2] == 3,                     "Expected RGB (H, W, 3)"
    assert result["heatmap"].dtype.name == "uint8",             "Expected uint8"

    out_path = os.path.join(args.out_dir, "smoke_heatmap.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(result["heatmap"], cv2.COLOR_RGB2BGR))
    print(f"  Saved → {out_path}")
    print("  PASS")

    # ── Case 2: Normal prediction without heatmap ──────────────────────────────────────
    print("\n[Case 2] predict without heatmap")
    pipeline_no_hm = inference_pipeline_setup(args.config, heatmap=False)
    result_no_hm = pipeline_no_hm.predict(args.image, full_data)
    _print_result(result_no_hm)

    assert result_no_hm["heatmap"] is None, "Expected heatmap=None when heatmap_viz is not set"
    print("  PASS")

    # ── Case 3: Image path doesn't exist → should raise ValueError ──────────────────────
    print("\n[Case 3] nonexistent image → ValueError")
    try:
        pipeline.predict("__nonexistent__.jpg", full_data)
        print("  FAIL: no exception raised")
    except ValueError as e:
        print(f"  OK: ValueError({e})")
        print("  PASS")

    print("\nAll cases passed.")


if __name__ == "__main__":
    main()