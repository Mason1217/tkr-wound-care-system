import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # [Fix] MPS not support bicubic interpolation (when training dino)

import sys
import argparse
import random
import torch
import numpy as np

from core import TKRTrainingRunner

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Global seed set to {seed}")

def parse_args():
    parser = argparse.ArgumentParser(description="TKR Wound Analysis Training Pipeline")
    
    parser.add_argument(
        "--base",
        type=str,
        default="config/tkr_base_config.yaml",
        help="Path to the base configuration file."
    )
    
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="Path to the experiment configuration file (overrides base)."
    )

    return parser.parse_args()

def main():
    seed_everything()

    args = parse_args()

    if args.exp and not args.exp.endswith(".yaml"):
        print(f"Error: Experiment config file '{args.exp}' must be a .yaml file.")
        sys.exit(1)

    print("="*60)
    print("🚀  Starting TKR Training Pipeline")
    print(f"    Base Config : {args.base}")
    print(f"    Exp Config  : {args.exp if args.exp else 'None (Running Base Only)'}")
    print("="*60)

    try:
        runner = TKRTrainingRunner(
            base_config_path=args.base,
            exp_config_path=args.exp
        )

        runner.run()

        print("\n✅  All training jobs finished successfully.")

    except Exception as e:
        print(f"\n❌  Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()