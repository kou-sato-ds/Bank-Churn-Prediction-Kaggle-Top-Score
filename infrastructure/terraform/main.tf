# AWSプロバイダーの設定
provider "aws" {
  region = "ap-northeast-1" 
}

# 1. データレイクとなるS3バケットの定義
resource "aws_s3_bucket" "project_data_lake" {
  bucket = "sato-bank-churn-data-lake-20260316" 

  tags = {
    Name        = "Bank Churn Prediction Data Lake"
    Environment = "Dev"
    Project     = "Bank-Churn-Prediction"
  }
}

# 2. バージョニング設定
resource "aws_s3_bucket_versioning" "v1" {
  bucket = aws_s3_bucket.project_data_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}