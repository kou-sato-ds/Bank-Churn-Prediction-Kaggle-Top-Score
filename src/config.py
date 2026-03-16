# src/config.py
AWS_REGION = "ap-northeast-1"
S3_BUCKET_NAME = "sato-bank-churn-data-lake-20260316"  # 作成した名前に合わせてください
RAW_DATA_PATH = "data/raw/train.csv" # ローカルの場所
S3_DATA_KEY = "input/train.csv"      # S3の中での名前