python tkr_train.py --exp config/exp015_ablation/seed40/training/overfit_test.yaml

python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_overfit_test --target results/exp015_ablation/exp015_overfit_test/inference_asset/models/multimodal   --model_file_name best_model.pth
python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_overfit_test --target results/exp015_ablation/exp015_overfit_test/inference_asset/processors          --model_file_name preprocessor.pkl
python -m tkr_inference.gather_models --source yolo/models/                                --target results/exp015_ablation/exp015_overfit_test/inference_asset/yolo                --model_file_name v1.pt

python tkr_analysis.py --config results/exp015_ablation/exp015_overfit_test/run1_split_versionseed_40_fold_0/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_overfit_test/run1_split_versionseed_40_fold_0/checkpoints/stage2_finetune/best_model.pth
                       --heatmap
python tkr_analysis.py --config results/exp015_ablation/exp015_overfit_test/run2_split_versionseed_40_fold_1/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_overfit_test/run2_split_versionseed_40_fold_1/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_overfit_test/run3_split_versionseed_40_fold_2/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_overfit_test/run3_split_versionseed_40_fold_2/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_overfit_test/run4_split_versionseed_40_fold_3/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_overfit_test/run4_split_versionseed_40_fold_3/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_overfit_test/run5_split_versionseed_40_fold_4/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_overfit_test/run5_split_versionseed_40_fold_4/checkpoints/stage2_finetune/best_model.pth
python -m analysis.cv_analysis \
       --base config/tkr_base_config.yaml \
       --exp config/exp015_ablation/seed40/training/overfit_test.yaml \
       --report checkpoints/stage2_finetune/analysis_results/threshold_0.5/classification_report_image_level.json

PYTORCH_ENABLE_MPS_FALLBACK=1 python -m tkr_inference.run_inference --inference_config config/exp015_ablation/seed40/inference/overfit_test.yaml --update_calib

no aug
python tkr_train.py --exp config/exp015_ablation/seed40/training/aug_test_none.yaml

python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_aug_test_none --target results/exp015_ablation/exp015_aug_test_none/inference_asset/models/multimodal   --model_file_name best_model.pth
python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_aug_test_none --target results/exp015_ablation/exp015_aug_test_none/inference_asset/processors          --model_file_name preprocessor.pkl
python -m tkr_inference.gather_models --source yolo/models/                                --target results/exp015_ablation/exp015_aug_test_none/inference_asset/yolo                --model_file_name v1.pt

python tkr_analysis.py --config results/exp015_ablation/exp015_aug_test_none/run1_split_versionseed_40_fold_0/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_aug_test_none/run1_split_versionseed_40_fold_0/checkpoints/stage2_finetune/best_model.pth
                       --heatmap
python tkr_analysis.py --config results/exp015_ablation/exp015_aug_test_none/run2_split_versionseed_40_fold_1/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_aug_test_none/run2_split_versionseed_40_fold_1/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_aug_test_none/run3_split_versionseed_40_fold_2/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_aug_test_none/run3_split_versionseed_40_fold_2/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_aug_test_none/run4_split_versionseed_40_fold_3/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_aug_test_none/run4_split_versionseed_40_fold_3/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_aug_test_none/run5_split_versionseed_40_fold_4/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_aug_test_none/run5_split_versionseed_40_fold_4/checkpoints/stage2_finetune/best_model.pth
python -m analysis.cv_analysis \
       --base config/tkr_base_config.yaml \
       --exp config/exp015_ablation/seed40/training/aug_test_none.yaml \
       --report checkpoints/stage2_finetune/analysis_results/threshold_0.5/classification_report_image_level.json


basic aug
python tkr_train.py --exp config/exp015_ablation/seed40/training/aug_test_basic.yaml

python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_aug_test_basic --target results/exp015_ablation/exp015_aug_test_basic/inference_asset/models/multimodal   --model_file_name best_model.pth
python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_aug_test_basic --target results/exp015_ablation/exp015_aug_test_basic/inference_asset/processors          --model_file_name preprocessor.pkl
python -m tkr_inference.gather_models --source yolo/models/                                --target results/exp015_ablation/exp015_aug_test_basic/inference_asset/yolo                --model_file_name v1.pt

python tkr_analysis.py --config results/exp015_ablation/exp015_aug_test_basic/run1_split_versionseed_40_fold_0/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_aug_test_basic/run1_split_versionseed_40_fold_0/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_aug_test_basic/run2_split_versionseed_40_fold_1/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_aug_test_basic/run2_split_versionseed_40_fold_1/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_aug_test_basic/run3_split_versionseed_40_fold_2/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_aug_test_basic/run3_split_versionseed_40_fold_2/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_aug_test_basic/run4_split_versionseed_40_fold_3/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_aug_test_basic/run4_split_versionseed_40_fold_3/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_aug_test_basic/run5_split_versionseed_40_fold_4/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_aug_test_basic/run5_split_versionseed_40_fold_4/checkpoints/stage2_finetune/best_model.pth
python -m analysis.cv_analysis \
       --base config/tkr_base_config.yaml \
       --exp config/exp015_ablation/seed40/training/aug_test_basic.yaml \
       --report checkpoints/stage2_finetune/analysis_results/threshold_0.5/classification_report_image_level.json

PYTORCH_ENABLE_MPS_FALLBACK=1 python -m tkr_inference.run_inference --inference_config config/exp015_ablation/seed40/inference/aug_test_basic.yaml --update_calib

original
python tkr_train.py --exp config/exp015_ablation/seed40/training/original.yaml

python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_original --target results/exp015_ablation/exp015_original/inference_asset/models/multimodal   --model_file_name best_model.pth
python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_original --target results/exp015_ablation/exp015_original/inference_asset/processors          --model_file_name preprocessor.pkl
python -m tkr_inference.gather_models --source yolo/models/                                --target results/exp015_ablation/exp015_original/inference_asset/yolo                --model_file_name v1.pt

python tkr_analysis.py --config results/exp015_ablation/exp015_original/run1_split_versionseed_40_fold_0/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_original/run1_split_versionseed_40_fold_0/checkpoints/stage2_finetune/best_model.pth
                       --heatmap
python tkr_analysis.py --config results/exp015_ablation/exp015_original/run2_split_versionseed_40_fold_1/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_original/run2_split_versionseed_40_fold_1/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_original/run3_split_versionseed_40_fold_2/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_original/run3_split_versionseed_40_fold_2/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_original/run4_split_versionseed_40_fold_3/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_original/run4_split_versionseed_40_fold_3/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_original/run5_split_versionseed_40_fold_4/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_original/run5_split_versionseed_40_fold_4/checkpoints/stage2_finetune/best_model.pth
python -m analysis.cv_analysis \
       --base config/tkr_base_config.yaml \
       --exp config/exp015_ablation/seed40/training/original.yaml \
       --report checkpoints/stage2_finetune/analysis_results/threshold_0.5/classification_report_image_level.json

PYTORCH_ENABLE_MPS_FALLBACK=1 python -m tkr_inference.run_inference --inference_config config/exp015_ablation/seed40/inference/original.yaml 

crop yolo
python tkr_train.py --exp config/exp015_ablation/seed40/training/crop_test_yolo.yaml

python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_crop_test_yolo --target results/exp015_ablation/exp015_crop_test_yolo/inference_asset/models/multimodal   --model_file_name best_model.pth
python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_crop_test_yolo --target results/exp015_ablation/exp015_crop_test_yolo/inference_asset/processors          --model_file_name preprocessor.pkl
python -m tkr_inference.gather_models --source yolo/models/                                --target results/exp015_ablation/exp015_crop_test_yolo/inference_asset/yolo                --model_file_name v1.pt

python tkr_analysis.py --config results/exp015_ablation/exp015_crop_test_yolo/run1_split_versionseed_40_fold_0/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_crop_test_yolo/run1_split_versionseed_40_fold_0/checkpoints/stage2_finetune/best_model.pth
                       --heatmap
python tkr_analysis.py --config results/exp015_ablation/exp015_crop_test_yolo/run2_split_versionseed_40_fold_1/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_crop_test_yolo/run2_split_versionseed_40_fold_1/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_crop_test_yolo/run3_split_versionseed_40_fold_2/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_crop_test_yolo/run3_split_versionseed_40_fold_2/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_crop_test_yolo/run4_split_versionseed_40_fold_3/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_crop_test_yolo/run4_split_versionseed_40_fold_3/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_crop_test_yolo/run5_split_versionseed_40_fold_4/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_crop_test_yolo/run5_split_versionseed_40_fold_4/checkpoints/stage2_finetune/best_model.pth
python -m analysis.cv_analysis \
       --base config/tkr_base_config.yaml \
       --exp config/exp015_ablation/seed40/training/crop_test_yolo.yaml \
       --report checkpoints/stage2_finetune/analysis_results/threshold_0.5/classification_report_image_level.json

PYTORCH_ENABLE_MPS_FALLBACK=1 python -m tkr_inference.run_inference --inference_config config/exp015_ablation/seed40/inference/crop_test_yolo.yaml --update_calib

image only
python tkr_train.py --exp config/exp015_ablation/seed40/training/input_test_image_only.yaml

python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_input_test_image_only --target results/exp015_ablation/exp015_input_test_image_only/inference_asset/models/multimodal   --model_file_name best_model.pth
python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_input_test_image_only --target results/exp015_ablation/exp015_input_test_image_only/inference_asset/processors          --model_file_name preprocessor.pkl
python -m tkr_inference.gather_models --source yolo/models/                                --target results/exp015_ablation/exp015_input_test_image_only/inference_asset/yolo                --model_file_name v1.pt

python tkr_analysis.py --config results/exp015_ablation/exp015_input_test_image_only/run1_split_versionseed_40_fold_0/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_input_test_image_only/run1_split_versionseed_40_fold_0/checkpoints/stage2_finetune/best_model.pth
                       --heatmap
python tkr_analysis.py --config results/exp015_ablation/exp015_input_test_image_only/run2_split_versionseed_40_fold_1/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_input_test_image_only/run2_split_versionseed_40_fold_1/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_input_test_image_only/run3_split_versionseed_40_fold_2/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_input_test_image_only/run3_split_versionseed_40_fold_2/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_input_test_image_only/run4_split_versionseed_40_fold_3/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_input_test_image_only/run4_split_versionseed_40_fold_3/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_input_test_image_only/run5_split_versionseed_40_fold_4/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_input_test_image_only/run5_split_versionseed_40_fold_4/checkpoints/stage2_finetune/best_model.pth
python -m analysis.cv_analysis \
       --base config/tkr_base_config.yaml \
       --exp config/exp015_ablation/seed40/training/input_test_image_only.yaml \
       --report checkpoints/stage2_finetune/analysis_results/threshold_0.5/classification_report_image_level.json

PYTORCH_ENABLE_MPS_FALLBACK=1 python -m tkr_inference.run_inference --inference_config config/exp015_ablation/seed40/inference/input_test_image_only.yaml --update_calib

vits16
python tkr_train.py --exp config/exp015_ablation/seed40/training/model_test_vits16.yaml

python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_model_test_vits16 --target results/exp015_ablation/exp015_model_test_vits16/inference_asset/models/multimodal   --model_file_name best_model.pth
python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_model_test_vits16 --target results/exp015_ablation/exp015_model_test_vits16/inference_asset/processors          --model_file_name preprocessor.pkl
python -m tkr_inference.gather_models --source yolo/models/                                --target results/exp015_ablation/exp015_model_test_vits16/inference_asset/yolo                --model_file_name v1.pt

python tkr_analysis.py --config results/exp015_ablation/exp015_model_test_vits16/run1_split_versionseed_40_fold_0/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_model_test_vits16/run1_split_versionseed_40_fold_0/checkpoints/stage2_finetune/best_model.pth
                       --heatmap
python tkr_analysis.py --config results/exp015_ablation/exp015_model_test_vits16/run2_split_versionseed_40_fold_1/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_model_test_vits16/run2_split_versionseed_40_fold_1/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_model_test_vits16/run3_split_versionseed_40_fold_2/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_model_test_vits16/run3_split_versionseed_40_fold_2/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_model_test_vits16/run4_split_versionseed_40_fold_3/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_model_test_vits16/run4_split_versionseed_40_fold_3/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_model_test_vits16/run5_split_versionseed_40_fold_4/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_model_test_vits16/run5_split_versionseed_40_fold_4/checkpoints/stage2_finetune/best_model.pth
python -m analysis.cv_analysis \
       --base config/tkr_base_config.yaml \
       --exp config/exp015_ablation/seed40/training/model_test_vits16.yaml \
       --report checkpoints/stage2_finetune/analysis_results/threshold_0.5/classification_report_image_level.json

PYTORCH_ENABLE_MPS_FALLBACK=1 python -m tkr_inference.run_inference --inference_config config/exp015_ablation/seed40/inference/model_test_vits16.yaml --update_calib

resnet50
python tkr_train.py --exp config/exp015_ablation/seed40/training/model_test_resnet50.yaml

python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_model_test_resnet50 --target results/exp015_ablation/exp015_model_test_resnet50/inference_asset/models/multimodal   --model_file_name best_model.pth
python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_model_test_resnet50 --target results/exp015_ablation/exp015_model_test_resnet50/inference_asset/processors          --model_file_name preprocessor.pkl
python -m tkr_inference.gather_models --source yolo/models/                                       --target results/exp015_ablation/exp015_model_test_resnet50/inference_asset/yolo                       --model_file_name v1.pt

python tkr_analysis.py --config results/exp015_ablation/exp015_model_test_resnet50/run1_split_versionseed_40_fold_0/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_model_test_resnet50/run1_split_versionseed_40_fold_0/checkpoints/stage2_finetune/best_model.pth
                       --heatmap
python tkr_analysis.py --config results/exp015_ablation/exp015_model_test_resnet50/run2_split_versionseed_40_fold_1/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_model_test_resnet50/run2_split_versionseed_40_fold_1/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_model_test_resnet50/run3_split_versionseed_40_fold_2/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_model_test_resnet50/run3_split_versionseed_40_fold_2/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_model_test_resnet50/run4_split_versionseed_40_fold_3/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_model_test_resnet50/run4_split_versionseed_40_fold_3/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_model_test_resnet50/run5_split_versionseed_40_fold_4/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_model_test_resnet50/run5_split_versionseed_40_fold_4/checkpoints/stage2_finetune/best_model.pth
python -m analysis.cv_analysis \
       --base config/tkr_base_config.yaml \
       --exp config/exp015_ablation/seed40/training/model_test_resnet50.yaml \
       --report checkpoints/stage2_finetune/analysis_results/threshold_0.5/classification_report_image_level.json

PYTORCH_ENABLE_MPS_FALLBACK=1 python -m tkr_inference.run_inference --inference_config config/exp015_ablation/seed40/inference/model_test_resnet50.yaml --update_calib

vitb16
python tkr_train.py --exp config/exp015_ablation/seed40/training/model_test_vitb16.yaml

python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_model_test_vitb16 --target results/exp015_ablation/exp015_model_test_vitb16/inference_asset/models/multimodal   --model_file_name best_model.pth
python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_model_test_vitb16 --target results/exp015_ablation/exp015_model_test_vitb16/inference_asset/processors          --model_file_name preprocessor.pkl
python -m tkr_inference.gather_models --source yolo/models/                                --target results/exp015_ablation/exp015_model_test_vitb16/inference_asset/yolo                --model_file_name v1.pt

python tkr_analysis.py --config results/exp015_ablation/exp015_model_test_vitb16/run1_split_versionseed_40_fold_0/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_model_test_vitb16/run1_split_versionseed_40_fold_0/checkpoints/stage2_finetune/best_model.pth
                       --heatmap
python tkr_analysis.py --config results/exp015_ablation/exp015_model_test_vitb16/run2_split_versionseed_40_fold_1/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_model_test_vitb16/run2_split_versionseed_40_fold_1/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_model_test_vitb16/run3_split_versionseed_40_fold_2/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_model_test_vitb16/run3_split_versionseed_40_fold_2/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_model_test_vitb16/run4_split_versionseed_40_fold_3/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_model_test_vitb16/run4_split_versionseed_40_fold_3/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_model_test_vitb16/run5_split_versionseed_40_fold_4/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_model_test_vitb16/run5_split_versionseed_40_fold_4/checkpoints/stage2_finetune/best_model.pth
python -m analysis.cv_analysis \
       --base config/tkr_base_config.yaml \
       --exp config/exp015_ablation/seed40/training/model_test_vitb16.yaml \
       --report checkpoints/stage2_finetune/analysis_results/threshold_0.5/classification_report_image_level.json

PYTORCH_ENABLE_MPS_FALLBACK=1 python -m tkr_inference.run_inference --inference_config config/exp015_ablation/seed40/inference/model_test_vitb16.yaml --update_calib

cutmix
python tkr_train.py --exp config/exp015_ablation/seed40/training/aug_test_cutmix.yaml

python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_aug_test_cutmix --target results/exp015_ablation/exp015_aug_test_cutmix/inference_asset/models/multimodal   --model_file_name best_model.pth
python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_aug_test_cutmix --target results/exp015_ablation/exp015_aug_test_cutmix/inference_asset/processors          --model_file_name preprocessor.pkl
python -m tkr_inference.gather_models --source yolo/models/                                --target results/exp015_ablation/exp015_aug_test_cutmix/inference_asset/yolo                --model_file_name v1.pt

python tkr_analysis.py --config results/exp015_ablation/exp015_aug_test_cutmix/run1_split_versionseed_40_fold_0/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_aug_test_cutmix/run1_split_versionseed_40_fold_0/checkpoints/stage2_finetune/best_model.pth
                       --heatmap
python tkr_analysis.py --config results/exp015_ablation/exp015_aug_test_cutmix/run2_split_versionseed_40_fold_1/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_aug_test_cutmix/run2_split_versionseed_40_fold_1/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_aug_test_cutmix/run3_split_versionseed_40_fold_2/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_aug_test_cutmix/run3_split_versionseed_40_fold_2/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_aug_test_cutmix/run4_split_versionseed_40_fold_3/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_aug_test_cutmix/run4_split_versionseed_40_fold_3/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_aug_test_cutmix/run5_split_versionseed_40_fold_4/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_aug_test_cutmix/run5_split_versionseed_40_fold_4/checkpoints/stage2_finetune/best_model.pth
python -m analysis.cv_analysis \
       --base config/tkr_base_config.yaml \
       --exp config/exp015_ablation/seed40/training/aug_test_cutmix.yaml \
       --report checkpoints/stage2_finetune/analysis_results/threshold_0.5/classification_report_image_level.json

PYTORCH_ENABLE_MPS_FALLBACK=1 python -m tkr_inference.run_inference --inference_config config/exp015_ablation/seed40/inference/aug_test_cutmix.yaml --update_calib

no crop
python tkr_train.py --exp config/exp015_ablation/seed40/training/crop_test_none.yaml

python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_crop_test_none --target results/exp015_ablation/exp015_crop_test_none/inference_asset/models/multimodal   --model_file_name best_model.pth
python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_crop_test_none --target results/exp015_ablation/exp015_crop_test_none/inference_asset/processors          --model_file_name preprocessor.pkl
python -m tkr_inference.gather_models --source yolo/models/                                --target results/exp015_ablation/exp015_crop_test_none/inference_asset/yolo                --model_file_name v1.pt

python tkr_analysis.py --config results/exp015_ablation/exp015_crop_test_none/run1_split_versionseed_40_fold_0/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_crop_test_none/run1_split_versionseed_40_fold_0/checkpoints/stage2_finetune/best_model.pth
                       --heatmap
python tkr_analysis.py --config results/exp015_ablation/exp015_crop_test_none/run2_split_versionseed_40_fold_1/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_crop_test_none/run2_split_versionseed_40_fold_1/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_crop_test_none/run3_split_versionseed_40_fold_2/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_crop_test_none/run3_split_versionseed_40_fold_2/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_crop_test_none/run4_split_versionseed_40_fold_3/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_crop_test_none/run4_split_versionseed_40_fold_3/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_crop_test_none/run5_split_versionseed_40_fold_4/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_crop_test_none/run5_split_versionseed_40_fold_4/checkpoints/stage2_finetune/best_model.pth
python -m analysis.cv_analysis \
       --base config/tkr_base_config.yaml \
       --exp config/exp015_ablation/seed40/training/crop_test_none.yaml \
       --report checkpoints/stage2_finetune/analysis_results/threshold_0.5/classification_report_image_level.json

PYTORCH_ENABLE_MPS_FALLBACK=1 python -m tkr_inference.run_inference --inference_config config/exp015_ablation/seed40/inference/crop_test_none.yaml # --heatmap

full fine-tune
python tkr_train.py --exp config/exp015_ablation/seed40/training/fine_tune_test_full.yaml

python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_fine_tune_test_full --target results/exp015_ablation/exp015_fine_tune_test_full/inference_asset/models/multimodal   --model_file_name best_model.pth
python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_fine_tune_test_full --target results/exp015_ablation/exp015_fine_tune_test_full/inference_asset/processors          --model_file_name preprocessor.pkl
python -m tkr_inference.gather_models --source yolo/models/                                --target results/exp015_ablation/exp015_fine_tune_test_full/inference_asset/yolo                --model_file_name v1.pt

python tkr_analysis.py --config results/exp015_ablation/exp015_fine_tune_test_full/run1_split_versionseed_40_fold_0/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_fine_tune_test_full/run1_split_versionseed_40_fold_0/checkpoints/stage2_finetune/best_model.pth
                       --heatmap
python tkr_analysis.py --config results/exp015_ablation/exp015_fine_tune_test_full/run2_split_versionseed_40_fold_1/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_fine_tune_test_full/run2_split_versionseed_40_fold_1/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_fine_tune_test_full/run3_split_versionseed_40_fold_2/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_fine_tune_test_full/run3_split_versionseed_40_fold_2/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_fine_tune_test_full/run4_split_versionseed_40_fold_3/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_fine_tune_test_full/run4_split_versionseed_40_fold_3/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_fine_tune_test_full/run5_split_versionseed_40_fold_4/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_fine_tune_test_full/run5_split_versionseed_40_fold_4/checkpoints/stage2_finetune/best_model.pth
python -m analysis.cv_analysis \
       --base config/tkr_base_config.yaml \
       --exp config/exp015_ablation/seed40/training/fine_tune_test_full.yaml \
       --report checkpoints/stage2_finetune/analysis_results/threshold_0.5/classification_report_image_level.json

PYTORCH_ENABLE_MPS_FALLBACK=1 python -m tkr_inference.run_inference --inference_config config/exp015_ablation/seed40/inference/fine_tune_test_full.yaml --update_calib

lora rank 8
python tkr_train.py --exp config/exp015_ablation/seed40/training/fine_tune_test_lora_r8.yaml

python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_fine_tune_test_lora_r8 --target results/exp015_ablation/exp015_fine_tune_test_lora_r8/inference_asset/models/multimodal   --model_file_name best_model.pth
python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_fine_tune_test_lora_r8 --target results/exp015_ablation/exp015_fine_tune_test_lora_r8/inference_asset/processors          --model_file_name preprocessor.pkl
python -m tkr_inference.gather_models --source yolo/models/                                --target results/exp015_ablation/exp015_fine_tune_test_lora_r8/inference_asset/yolo                --model_file_name v1.pt

python tkr_analysis.py --config results/exp015_ablation/exp015_fine_tune_test_lora_r8/run1_split_versionseed_40_fold_0/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_fine_tune_test_lora_r8/run1_split_versionseed_40_fold_0/checkpoints/stage2_finetune/best_model.pth
                       --heatmap
python tkr_analysis.py --config results/exp015_ablation/exp015_fine_tune_test_lora_r8/run2_split_versionseed_40_fold_1/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_fine_tune_test_lora_r8/run2_split_versionseed_40_fold_1/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_fine_tune_test_lora_r8/run3_split_versionseed_40_fold_2/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_fine_tune_test_lora_r8/run3_split_versionseed_40_fold_2/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_fine_tune_test_lora_r8/run4_split_versionseed_40_fold_3/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_fine_tune_test_lora_r8/run4_split_versionseed_40_fold_3/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_fine_tune_test_lora_r8/run5_split_versionseed_40_fold_4/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_fine_tune_test_lora_r8/run5_split_versionseed_40_fold_4/checkpoints/stage2_finetune/best_model.pth
python -m analysis.cv_analysis \
       --base config/tkr_base_config.yaml \
       --exp config/exp015_ablation/seed40/training/fine_tune_test_lora_r8.yaml \
       --report checkpoints/stage2_finetune/analysis_results/threshold_0.5/classification_report_image_level.json

PYTORCH_ENABLE_MPS_FALLBACK=1 python -m tkr_inference.run_inference --inference_config config/exp015_ablation/seed40/inference/fine_tune_test_lora_r8.yaml --update_calib

ce
python tkr_train.py --exp config/exp015_ablation/seed40/training/loss_test_ce.yaml

python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_loss_test_ce --target results/exp015_ablation/exp015_loss_test_ce/inference_asset/models/multimodal   --model_file_name best_model.pth
python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_loss_test_ce --target results/exp015_ablation/exp015_loss_test_ce/inference_asset/processors          --model_file_name preprocessor.pkl
python -m tkr_inference.gather_models --source yolo/models/                                --target results/exp015_ablation/exp015_loss_test_ce/inference_asset/yolo                --model_file_name v1.pt

python tkr_analysis.py --config results/exp015_ablation/exp015_loss_test_ce/run1_split_versionseed_40_fold_0/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_loss_test_ce/run1_split_versionseed_40_fold_0/checkpoints/stage2_finetune/best_model.pth
                       --heatmap
python tkr_analysis.py --config results/exp015_ablation/exp015_loss_test_ce/run2_split_versionseed_40_fold_1/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_loss_test_ce/run2_split_versionseed_40_fold_1/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_loss_test_ce/run3_split_versionseed_40_fold_2/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_loss_test_ce/run3_split_versionseed_40_fold_2/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_loss_test_ce/run4_split_versionseed_40_fold_3/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_loss_test_ce/run4_split_versionseed_40_fold_3/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_loss_test_ce/run5_split_versionseed_40_fold_4/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_loss_test_ce/run5_split_versionseed_40_fold_4/checkpoints/stage2_finetune/best_model.pth
python -m analysis.cv_analysis \
       --base config/tkr_base_config.yaml \
       --exp config/exp015_ablation/seed40/training/loss_test_ce.yaml \
       --report checkpoints/stage2_finetune/analysis_results/threshold_0.5/classification_report_image_level.json

PYTORCH_ENABLE_MPS_FALLBACK=1 python -m tkr_inference.run_inference --inference_config config/exp015_ablation/seed40/inference/loss_test_ce.yaml --update_calib

focal loss
python tkr_train.py --exp config/exp015_ablation/seed40/training/loss_test_focal.yaml

python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_loss_test_focal --target results/exp015_ablation/exp015_loss_test_focal/inference_asset/models/multimodal   --model_file_name best_model.pth
python -m tkr_inference.gather_models --source results/exp015_ablation/exp015_loss_test_focal --target results/exp015_ablation/exp015_loss_test_focal/inference_asset/processors          --model_file_name preprocessor.pkl
python -m tkr_inference.gather_models --source yolo/models/                                --target results/exp015_ablation/exp015_loss_test_focal/inference_asset/yolo                --model_file_name v1.pt

python tkr_analysis.py --config results/exp015_ablation/exp015_loss_test_focal/run1_split_versionseed_40_fold_0/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_loss_test_focal/run1_split_versionseed_40_fold_0/checkpoints/stage2_finetune/best_model.pth
                       --heatmap
python tkr_analysis.py --config results/exp015_ablation/exp015_loss_test_focal/run2_split_versionseed_40_fold_1/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_loss_test_focal/run2_split_versionseed_40_fold_1/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_loss_test_focal/run3_split_versionseed_40_fold_2/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_loss_test_focal/run3_split_versionseed_40_fold_2/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_loss_test_focal/run4_split_versionseed_40_fold_3/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_loss_test_focal/run4_split_versionseed_40_fold_3/checkpoints/stage2_finetune/best_model.pth
python tkr_analysis.py --config results/exp015_ablation/exp015_loss_test_focal/run5_split_versionseed_40_fold_4/config.yaml \
                       --ckpt   results/exp015_ablation/exp015_loss_test_focal/run5_split_versionseed_40_fold_4/checkpoints/stage2_finetune/best_model.pth
python -m analysis.cv_analysis \
       --base config/tkr_base_config.yaml \
       --exp config/exp015_ablation/seed40/training/loss_test_focal.yaml \
       --report checkpoints/stage2_finetune/analysis_results/threshold_0.5/classification_report_image_level.json

PYTORCH_ENABLE_MPS_FALLBACK=1 python -m tkr_inference.run_inference --inference_config config/exp015_ablation/seed40/inference/loss_test_focal.yaml --update_calib
