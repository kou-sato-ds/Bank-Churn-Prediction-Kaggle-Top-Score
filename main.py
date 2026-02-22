"""
Bank Churn Prediction Model - Master Edition
Author: henohenomohezi
Kaggle Private Score: 0.93420
Description: 5-fold Stratified K-fold with LightGBM. 
Refactored for production-ready structure with logging and class-based feature engineering.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# ========================================== 
# 0. Setup: ログの設定（実務での信頼性を担保）
# ========================================== 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========================================== 
# 1. Feature Engineering: 職人の技をクラスに封じ込める
# ========================================== 
class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.cat_cols = ['Geography', 'Gender']

    def fit_transform(self, df):
        logger.info("Feature engineering started...")
        df = df.copy()
        
        # 【最強変数】年齢と商品数の比率（あなたの発見したKey Feature）
        df['Age_per_Product'] = df['Age'] / (df['NumOfProducts'] + 1e-5)
        
        # カテゴリ変数のエンコーディング
        for col in self.cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            
        return df

    def transform(self, df):
        df = df.copy()
        df['Age_per_Product'] = df['Age'] / (df['NumOfProducts'] + 1e-5)
        for col in self.cat_cols:
            le = self.label_encoders[col]
            df[col] = le.transform(df[col].astype(str))
        return df

# ========================================== 
# 2. Main Process: 学習と予測の実行
# ========================================== 
def main():
    # --- Read ---
    logger.info("Loading data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # --- Features ---
    fe = FeatureEngineer()
    train = fe.fit_transform(train)
    test = fe.transform(test)

    drop_cols = ['id', 'CustomerId', 'Surname', 'Exited']
    features = [c for c in train.columns if c not in drop_cols]
    
    X, y = train[features], train['Exited']
    test_X = test[features]

    # --- K-fold Strategy ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    params = {
        'objective': 'binary', 
        'metric': 'auc', 
        'verbosity': -1, 
        'learning_rate': 0.05,
        'random_state': 42
    }

    # --- Split & Train ---
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    logger.info("Starting 5-fold Stratified K-fold training...")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            params, dtrain, valid_sets=[dval],
            callbacks=[lgb.early_stopping(100)]
        )
        
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(test_X) / 5
        
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        logger.info(f"Fold {fold+1} AUC: {fold_auc:.5f}")

    # --- Submit & Evaluation ---
    final_score = roc_auc_score(y, oof_preds)
    logger.info(f"--- Final OOF AUC Score: {final_score:.5f} ---")

    submission = pd.DataFrame({'id': test['id'], 'Exited': test_preds})
    submission.to_csv('submission_master.csv', index=False)
    logger.info("Submission file saved as 'submission_master.csv'")

if __name__ == "__main__":
    main()