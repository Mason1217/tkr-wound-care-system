import os
import json
import argparse
import itertools
import pandas as pd
from sklearn.metrics import fbeta_score

from core import TKRTrainingRunner

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True, help="path to base config")
    parser.add_argument("--exp", type=str, required=True, help="path to experiment config")
    parser.add_argument("--report", type=str, required=True, help="pattern of path to json classification report from fold directory")
    
    return parser.parse_args()

def get_fold_dirs(runner: TKRTrainingRunner) -> list[str]:
    sweep_cfg = runner.cfg_manager.cfg.get("sweep", None)
    fold_dirs = []

    if sweep_cfg is not None:
        param_keys = list(sweep_cfg.keys())
        param_values = list(sweep_cfg.values())
        combinations = list(itertools.product(*param_values))
        
        for i, combo in enumerate(combinations):            
            fold_name = runner._generate_run_name(i + 1, param_keys, combo)
            fold_dir = os.path.join(runner.root_output_dir, fold_name)
            fold_dirs.append(fold_dir)

    return fold_dirs

def agg_cv_results(fold_dirs: list[str], report_path_pattern: str):
    results = []

    for i, fold_dir in enumerate(fold_dirs):
        report_path = os.path.join(fold_dir, report_path_pattern)
        
        if not os.path.exists(report_path):
            print(f"[Warning] Missing report for fold {i}: {report_path}")
            continue

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        abnormal_precision   = data["異常需醫療介入"]["precision"]
        abnormal_recall      = data["異常需醫療介入"]["recall"]
        abnormal_f2score     = 0.0
        if ((2**2 * abnormal_precision) + abnormal_recall) != 0:
            abnormal_f2score = (1 + 2**2) * (abnormal_precision * abnormal_recall) / ((2**2 * abnormal_precision) + abnormal_recall)
        
        results.append({
            "fold": f"Fold: {i}",
            "Abnormal_Recall (%)": abnormal_recall * 100,
            "Abnormal_F2 (%)": abnormal_f2score * 100,
            "Abnormal_Precision (%)": abnormal_precision * 100,
            "Normal_Recall (%)": data["觀察與冰敷組"]["recall"] * 100,
            "Accuracy (%)": data["accuracy"] * 100,
            "Macro_F1 (%)": data["macro avg"]["f1-score"] * 100,
        })

    if len(results):
        df = pd.DataFrame(results)
        summary = {
            "fold"                  : "Mean ± Std",
            "Abnormal_Recall (%)"   : f"{df['Abnormal_Recall (%)'].mean():.2f} ± {df['Abnormal_Recall (%)'].std():.2f}",
            "Abnormal_F2 (%)"       : f"{df['Abnormal_F2 (%)'].mean():.2f} ± {df['Abnormal_F2 (%)'].std():.2f}",
            "Abnormal_Precision (%)": f"{df['Abnormal_Precision (%)'].mean():.2f} ± {df['Abnormal_Precision (%)'].std():.2f}",
            "Normal_Recall (%)"     : f"{df['Normal_Recall (%)'].mean():.2f} ± {df['Normal_Recall (%)'].std():.2f}",
            "Accuracy (%)"          : f"{df['Accuracy (%)'].mean():.2f} ± {df['Accuracy (%)'].std():.2f}",
            "Macro_F1 (%)"          : f"{df['Macro_F1 (%)'].mean():.2f} ± {df['Macro_F1 (%)'].std():.2f}",
        }

        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

        save_dir = os.path.dirname(fold_dirs[0])
        save_path = os.path.join(save_dir, "cv_report.csv")

        df.to_csv(save_path, index=False, encoding="utf-8-sig")

def main():
    args = parse_args()
    base_cfg_path = args.base
    exp_cfg_path = args.exp
    report_path_pattern = args.report

    runner = TKRTrainingRunner(base_cfg_path, exp_cfg_path)
    fold_dirs = get_fold_dirs(runner)
    agg_cv_results(fold_dirs, report_path_pattern)

if __name__ == "__main__":
    main()