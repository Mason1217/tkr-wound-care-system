MASTER_METADATA_PATH="data/TKR/processed/master_metadata.csv"

python -m data_preprocessing.data_engineering.show_data_distribution --metadata "$MASTER_METADATA_PATH"