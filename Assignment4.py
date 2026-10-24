import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

RANDOM_STATE = 409

df = pd.read_csv('phone_trade_in.csv')
X = df.drop(['phone_id', 'resale_price'], axis=1)
y = df['resale_price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

num_features = ['release_year', 'days_since_launch', 'launch_price',
                'storage_gb', 'battery_health']
cat_features = ['brand', 'tier', 'screen_condition', 'has_box', 'repaired_before']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

missing_by_brand = X_train.groupby('brand')['battery_health'].apply(
    lambda x: x.isna().mean()
).sort_values(ascending=False)

X_train_age = X_train.copy()
X_train_age['phone_age'] = 2025 - X_train_age['release_year']
battery_brand_median = X_train.groupby('brand')['battery_health'].median()
battery_age_median = X_train_age.groupby('phone_age')['battery_health'].median()

print("各品牌电池健康缺失比例：")
print(missing_by_brand.round(3))
print("\n各品牌电池健康中位数：")
print(battery_brand_median.round(1))
print("\n各机龄电池健康中位数：")
print(battery_age_median.round(1))

plt.figure(figsize=(10, 5))
plt.scatter(X_train['days_since_launch'], y_train, alpha=0.3, s=10)
plt.xlabel('Days since launch')
plt.ylabel('Resale price ($)')
plt.title('Resale price vs. days since launch')
plt.show()


# 定义评估指标计算函数
def score_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return pd.Series({'RMSE': rmse, 'MAE': mae, 'R²': r2})


# ---------------------- Baseline 基线模型（新增） ----------------------
baseline_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('dummy', DummyRegressor(strategy="mean"))
])
baseline_pipeline.fit(X_train, y_train)
y_pred_baseline = baseline_pipeline.predict(X_test)
baseline_scores = score_model(y_test, y_pred_baseline)
print("Baseline（均值基线）得分：")
print(baseline_scores.round(2))
# ----------------------------------------------------------------------


lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('lin_reg', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
lr_scores = score_model(y_test, y_pred_lr)
print("线性回归得分：")
print(lr_scores.round(2))


sgd_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('sgd_reg', SGDRegressor(random_state=RANDOM_STATE))
])
sgd_param_grid = {
    'sgd_reg__max_iter': [10000, 100000],
    'sgd_reg__tol': [1e-5, 1e-3],
    'sgd_reg__eta0': [0.01, 0.1]
}
sgd_grid = GridSearchCV(
    sgd_pipeline, sgd_param_grid,
    cv=5, scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
sgd_grid.fit(X_train, y_train)
print("SGD最优参数：", sgd_grid.best_params_)
y_pred_sgd = sgd_grid.best_estimator_.predict(X_test)
sgd_scores = score_model(y_test, y_pred_sgd)
print("\nSGD最优模型得分：")
print(sgd_scores.round(2))


rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(random_state=RANDOM_STATE))
])
rf_param_grid = {
    'rf__n_estimators': [100, 150],
    'rf__max_depth': [5, 10]
}
rf_grid = GridSearchCV(
    rf_pipeline, rf_param_grid,
    cv=5, scoring='neg_root_mean_squared_error',
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)
print("随机森林最优参数：", rf_grid.best_params_)
y_pred_rf = rf_grid.best_estimator_.predict(X_test)
rf_scores = score_model(y_test, y_pred_rf)
print("\n随机森林最优模型得分：")
print(rf_scores.round(2))


comparison = pd.DataFrame({
    'Baseline': baseline_scores,
    'Linear Regression': lr_scores,
    'SGD': sgd_scores,
    'Random Forest': rf_scores
}).T.round(2)
print("所有模型得分对比：")
print(comparison)


# 统计线性回归预测负价格的数量
negative_pred = (y_pred_lr < 0).sum()
print(f"线性回归预测负价格的手机数量：{negative_pred}")


# 获取特征名
feature_names = lr_pipeline['preprocessor'].get_feature_names_out()
weights = pd.Series(lr_pipeline['lin_reg'].coef_, index=feature_names)

# 提取关键对比项
cracked_diff = weights['cat__screen_condition_cracked'] - weights['cat__screen_condition_perfect']
apple_honor_diff = weights['cat__brand_Apple'] - weights['cat__brand_Honor']
repaired_cost = weights['cat__repaired_before_True']
print(f"碎屏相对完美屏的价格差：${cracked_diff:.0f}")
print(f"苹果相对荣耀的溢价：${apple_honor_diff:.0f}")
print(f"维修过的价格损失：${repaired_cost:.0f}")


# 线性回归排列重要性
lr_perm = permutation_importance(
    lr_pipeline, X_test, y_test,
    scoring='neg_root_mean_squared_error',
    n_repeats=10, random_state=RANDOM_STATE
)
lr_importance = pd.Series(lr_perm.importances_mean, index=X_test.columns).sort_values(ascending=False)

# 随机森林排列重要性
rf_perm = permutation_importance(
    rf_grid.best_estimator_, X_test, y_test,
    scoring='neg_root_mean_squared_error',
    n_repeats=10, random_state=RANDOM_STATE
)
rf_importance = pd.Series(rf_perm.importances_mean, index=X_test.columns).sort_values(ascending=False)

print("线性回归 Top 5 重要特征：")
print(lr_importance.head().round(2))
print("\n随机森林 Top 5 重要特征：")
print(rf_importance.head().round(2))


# 作业给定的利润计算函数
def trade_in_profit(predicted, actual, offer_rate):
    offer = offer_rate * predicted
    accepted = offer >= 0.75 * actual
    profit = (actual - offer)[accepted]
    return accepted.sum(), profit.sum()


# 测试不同报价比例
offer_rates = np.arange(0.60, 0.91, 0.05)
profit_results = []
for rate in offer_rates:
    base_n, base_profit = trade_in_profit(y_pred_baseline, y_test.values, rate)
    lr_n, lr_profit = trade_in_profit(y_pred_lr, y_test.values, rate)
    rf_n, rf_profit = trade_in_profit(y_pred_rf, y_test.values, rate)
    profit_results.append({
        'offer_rate': round(rate, 2),
        'baseline_phones': base_n,
        'baseline_profit': base_profit,
        'lr_phones': lr_n,
        'lr_profit': lr_profit,
        'rf_phones': rf_n,
        'rf_profit': rf_profit
    })
profit_df = pd.DataFrame(profit_results).set_index('offer_rate')
print("不同报价比例下的收购量和总利润：")
print(profit_df.round(0))


# 线性回归 0.75 报价下按机龄的利润分布
X_test_age = X_test.copy()
X_test_age['phone_age'] = 2025 - X_test_age['release_year']
X_test_age['actual'] = y_test.values
X_test_age['pred_lr'] = y_pred_lr
offer_rate = 0.75
offer = offer_rate * y_pred_lr
accepted = offer >= 0.75 * y_test.values
X_test_age['profit'] = np.where(accepted, y_test.values - offer, 0)
profit_by_age = X_test_age.groupby('phone_age')['profit'].sum()
print("线性回归 0.75 报价下各机龄总利润：")
print(profit_by_age.round(0))
