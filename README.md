# Bank Churn Prediction (Kaggle Private Score: 0.93420)

Kaggle Playground Series (Season 4, Episode 1) の銀行顧客離脱予測コンペティションにおいて、**Private Score 0.93420**（上位相当）を達成した解法リポジトリです。

## 📊 成果 (Results)
- **Private Score**: 0.93420
- **Public Score**: 0.93278
- **Model**: LightGBM
- **Evaluation**: 5-fold Stratified K-Fold (OOF)

## 💡 こだわりのポイント (Key Insights)

### 1. 特徴量エンジニアリング (Feature Engineering)
ビジネスドメインの視点から、顧客のロイヤリティを可視化する **`Age_per_Product`** (Age / NumOfProducts) を考案。この変数が最も高い予測寄与度（Gain）を示し、精度の向上に大きく貢献しました。

### 2. 一般化性能の確保 (Generalization)
実務での運用を想定し、単一の検証ではなく **OOF (Out-of-Fold) 予測** を採用。一度も学習に使われていないデータに対する評価を繰り返すことで、PublicよりもPrivateスコアが高い（未知のデータに強い）堅牢なモデルを実現しました。

### 3. 保守性の高いコード構造 (Code Quality)
前処理を関数化（`preprocess`）し、学習（Train）と推論（Test）のパイプラインを完全に同期。データセットの不整合によるエラーを未然に防ぐ実務的な実装を徹底しました。

## 🛠️ 使い方 (How to use)

### セットアップ
```bash
pip install -r requirements.txt
