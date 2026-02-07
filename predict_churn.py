"""
Bank Churn Prediction Model
Author: henohenomohezi
Kaggle Private Score: 0.93420
Description: 5-fold Stratified K-fold with LightGBM and custom feature engineering.
"""

# ========================================== 
# 1. Import: 道具を揃える
# ========================================== 
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# ========================================== 
# 2. Read: データの読み込み
# ========================================== 
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# ========================================== 
# 3. Features: 職人技の変数作成
# ========================================== 
def preprocess(df):
    # 【最強変数】年齢と商品数の比率（Gain値トップ）
    df['Age_per_Product'] = df['Age'] / (df['NumOfProducts'] + 1e-5)
    
    # カテゴリ変数を数値に変換
    for col in ['Geography', 'Gender']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

train = preprocess(train)
test = preprocess(test)

# 不要な列を削り、特徴量リストを確定
drop_cols = ['id', 'CustomerId', 'Surname', 'Exited']
features = [c for c in train.columns if c not in drop_cols]

# ========================================== 
# 4. K-fold Strategy: 鉄壁の5分割
# ========================================== 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
params = {
    'objective': 'binary', 
    'metric': 'auc', 
    'verbosity': -1, 
    'learning_rate': 0.05,
    'random_state': 42
}

# ========================================== 
# 5. Split & Train: OOFとアンサンブル予測
# ========================================== 
X, y = train[features], train['Exited']
test_X = test[features]

oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(params, dtrain, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100)])
    
    # OOF（Foldの外）の予測：これが手元の信頼スコアになる
    oof_preds[val_idx] = model.predict(X_val)
    # テスト予測の加算：平均を取って安定させる
    test_preds += model.predict(test_X) / 5

# ========================================== 
# 6. Submit: 成果の評価と出力
# ========================================== 
# OOFスコアの確認（これが本番スコアの予知夢！）
score = roc_auc_score(y, oof_preds)
print(f"--- Final OOF AUC: {score:.5f} ---")

submission = pd.DataFrame({'id': test['id'], 'Exited': test_preds})
submission.to_csv('submission_v11.csv', index=False)