import sys
import shutil
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source"          , required=True, help="Absolute or relative path to the directory contains results of all folds")
    parser.add_argument("--target"          , required=True, help="Absolute or relative path to the target directory for models for inference")
    parser.add_argument("--model_file_name" , required=True, help="file name e.g. best_model.pth, preprocessor.pkl, etc.")

    return parser.parse_args()

def find_all_models(source: str, target: str, model_file: str):
    source_dir = Path(source).resolve()
    target_dir = Path(target).resolve()

    if not source_dir.exists():
        print(f"[Gather Models]\t [Error]\t {source_dir} doesn't exist.")
        sys.exit(1)

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Gather Models]\t found models will be saved to {target_dir}")

    if model_file == "best_model.pth":
        _gather_best_model(source_dir, target_dir, model_file)

    elif model_file == "preprocessor.pkl":
        _gather_processors(source_dir, target_dir, model_file)
    
    else:
        _gather_models(source_dir, target_dir, model_file)


def _gather_models(source_dir, target_dir, model_file):
    count = 0

    for model_path in source_dir.rglob(model_file):
        target_filepath = target_dir / model_file
        shutil.copy2(model_path, target_filepath)
        print(f"[Gather Models]\t copied\n\t {model_path} \n\tto\n\t {target_filepath}\n")
        count += 1

    print(f"[Gather Models]\t found {count} models and saved to {target_dir}")

def _gather_best_model(source_dir, target_dir, model_file):
    count = 0

    for model_path in source_dir.rglob(model_file):
        parts = model_path.parts

        fold_info = next((p for p in parts if "run" in p or "fold" in p), "unknown_run")
        stage_info = next((p for p in parts if "stage" in p), "unknown_stage")

        if "stage1" in stage_info: continue

        target_filename = f"{fold_info}_{stage_info}_best_model.pth"
        target_filepath = target_dir / target_filename

        shutil.copy2(model_path, target_filepath)
        print(f"[Gather Models]\t copied\n\t {model_path} \n\tto\n\t {target_filepath}\n")
        count += 1

    print(f"[Gather Models]\t found {count} models and saved to {target_dir}")

def _gather_processors(source_dir, target_dir, model_file):
    count = 0

    for model_path in source_dir.rglob(model_file):
        parts = model_path.parts

        fold_info = next((p for p in parts if "run" in p or "fold" in p), "unknown_run")

        target_filename = f"{fold_info}_tabular_processor.pkl"
        target_filepath = target_dir / target_filename

        shutil.copy2(model_path, target_filepath)
        print(f"[Gather Models]\t copied\n\t {model_path} \n\tto\n\t {target_filepath}\n")
        count += 1

    print(f"[Gather Models]\t found {count} processors and saved to {target_dir}")

def main():
    args = parse_args()
    find_all_models(args.source, args.target, args.model_file_name)

if __name__ == "__main__":
    main()
