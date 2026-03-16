# 🚀 Bank Churn Prediction (Kaggle Private Score: 0.93420)

Kaggle Playground Series (Season 4, Episode 1) の銀行顧客離脱予測コンペティションにおいて、**Private Score 0.93420** を達成した解法リポジトリです。
本プロジェクトでは、高精度な予測モデルの構築に加え、**TerraformによるIaC（Infrastructure as Code）を用いたAWSデータレイクの自動構築**を統合し、実務レベルのMLOpsパイプラインを実装しています。

---

## 📊 成果 (Results)
- **Private Score**: 0.93420
- **Public Score**: 0.93278
- **Model**: LightGBM
- **Infrastructure**: AWS S3 managed by Terraform

---

## 🛠️ エンジニアリング・ハイライト (Engineering Highlights)

### 1. IaC によるデータ基盤の自動構築
`infrastructure/terraform` にて、データレイクとなるAWS S3バケットの構成をコード化。環境の再現性（冪等性）を担保し、バージョニング設定によりデータの堅牢性を確保しています。

#### **証跡：Terraformによるリソース作成成功**
![Terraform Apply](docs\images\terraform_apply.png)

### 2. クラウド・データパイプラインの実装
Python (`boto3`) を用いて、ローカルの分析資産をクラウドへ自動搬送するパイプラインを実装。プロジェクトルートを起点としたモジュール実行により、保守性の高いコードを実現しました。

#### **証跡：データアップロードの完遂とクラウド上での確認**
![Upload Success](docs\images\upload_success.png)
![S3 Console](docs\images\s3_console.png)

---

## 💡 分析のこだわり (Key Insights)

### 1. 特徴量エンジニアリング
ビジネスドメインの視点から、顧客のロイヤリティを可視化する **`Age_per_Product`** (Age / NumOfProducts) を考案。この変数が最も高い予測寄与度（Gain）を示し、精度の向上に大きく貢献しました。

### 2. 一般化性能の確保
実務での運用を想定し、**5-fold Stratified K-Fold (OOF)** を採用。未知のデータに対する堅牢性を追求し、PublicよりもPrivateスコアが高い結果（堅牢なモデル）を得ました。

---

## 📂 プロジェクト構造 (Directory Structure)
実務のMLOps環境に準拠し、実験用ノートブック、学習ソースコード、インフラ定義を明確に分離しています。

#### **証跡：整理されたディレクトリ構成**
![Project Tree](docs\images\project_tree.png)

```text
.
├── data/           # ローカルデータ (Git ignore)
├── docs/           # ドキュメント・スクリーンショット
├── infrastructure/ # Terraform (AWS構成)
├── notebooks/      # 実験用Jupyter Notebooks
├── src/            # 学習・推論・転送ロジック (Python modules)
└── main.py         # 実行エントリーポイント

##🛠️ 使い方 (How to use)
1. セットアップ
リポジトリをクローン後、必要なライブラリをインストールします。

Bash
pip install -r requirements.txt
2. インフラ構築 (AWS S3)
Terraformを使用して、クラウド上にデータレイク（S3バケット）を作成します。

Bash
cd infrastructure/terraform
terraform init
terraform apply -auto-approve
3. データアップロード
Pythonスクリプトを実行し、ローカルデータをS3へ転送します。

Bash
# プロジェクトルートへ移動してから実行
cd ../../
python -m src.upload_data
4. インフラの破棄
使用後は、不要なコストを防ぐためにリソースを削除します。

Bash
cd infrastructure/terraform
terraform destroy -auto-approve