python -m data_preprocessing.data_engineering.builder --raw_img_dir_names 工作表1 segmentation/07 segmentation/08 segmentation/09 segmentation/10 segmentation/11 --img_map_filename image_id_map.json
python -m data_preprocessing.data_engineering.splitter --processed_dir data/TKR/processed --img_map_filename image_id_map.json --metadata_filename master_metadata.csv --split_version 0325_cv
