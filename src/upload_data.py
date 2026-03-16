import boto3
import os
from src.config import S3_BUCKET_NAME, RAW_DATA_PATH, S3_DATA_KEY

def main():
    # S3クライアントの初期化
    s3 = boto3.client('s3')
    
    # ローカルファイルの存在確認（念のため）
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: File not found at {RAW_DATA_PATH}")
        return

    try:
        print(f"Uploading {RAW_DATA_PATH} to S3 bucket: {S3_BUCKET_NAME}...")
        # ファイルのアップロード実行
        s3.upload_file(RAW_DATA_PATH, S3_BUCKET_NAME, S3_DATA_KEY)
        print("Upload Successful! 🚀 Data is now in the cloud.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()