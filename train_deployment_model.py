# train_deployment_model.py
import pandas as pd
import lightgbm as lgb
import joblib # 用于保存模型和映射

print("正在为Web应用训练专属的部署模型...")

TARGET = 'accident_risk'
train_df = pd.read_csv('data/train.csv')

# 对所有类别特征进行编码，并保存映射
cat_features = ['road_type', 'lighting', 'weather', 'time_of_day']
mappings = {}
for col in cat_features:
    train_df[col], mapping = pd.factorize(train_df[col])
    mappings[col] = {label: i for i, label in enumerate(mapping)}

joblib.dump(mappings, 'category_mappings.joblib') # 保存映射

# 准备训练数据
features = [col for col in train_df.columns if col not in ['id', TARGET]]
X, y = train_df[features], train_df[TARGET]

# 使用我们的“黄金参数”进行训练
LGBM_PARAMS = {
    'random_state': 42, 'n_estimators': 500, 'objective': 'regression_l1', 'metric': 'rmse', 
    'learning_rate': 0.02, 'num_leaves': 31, 'max_depth': 7
}
model = lgb.LGBMRegressor(**LGBM_PARAMS)
model.fit(X, y)

joblib.dump(model, 'deployment_lgbm.joblib') # 保存模型

print("部署模型 'deployment_lgbm.joblib' 和映射文件 'category_mappings.joblib' 已成功保存！")