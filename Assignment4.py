import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
import gc

st.set_page_config(page_title="Phone Trade Model", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('phone_trade_in.csv')
    return df

RANDOM_STATE = 409
df = load_data()

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

st.subheader("各品牌电池健康缺失比例")
st.write(missing_by_brand.round(3))
st.subheader("各品牌电池健康中位数")
st.write(battery_brand_median.round(1))
st.subheader("各机龄电池健康中位数")
st.write(battery_age_median.round(1))

def score_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return pd.Series({'RMSE': rmse, 'MAE': mae, 'R²': r2})

# Baseline
baseline_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('dummy', DummyRegressor(strategy="mean"))
])
baseline_pipeline.fit(X_train, y_train)
y_pred_baseline = baseline_pipeline.predict(X_test)
baseline_scores = score_model(y_test, y_pred_baseline)
st.subheader("Baseline（均值基线）得分")
st.write(baseline_scores.round(2))
del baseline_pipeline
gc.collect()

# Linear Regression
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('lin_reg', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
lr_scores = score_model(y_test, y_pred_lr)
st.subheader("线性回归得分")
st.write(lr_scores.round(2))

# SGD 关掉网格搜索，直接训练，不再GridSearchCV（最大内存杀手直接砍掉）
sgd_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('sgd_reg', SGDRegressor(max_iter=10000, tol=1e-5, eta0=0.01, random_state=RANDOM_STATE))
])
sgd_pipeline.fit(X_train, y_train)
y_pred_sgd = sgd_pipeline.predict(X_test)
sgd_scores = score_model(y_test, y_pred_sgd)
st.subheader("SGD模型得分")
st.write(sgd_scores.round(2))
del sgd_pipeline
gc.collect()

# RandomForest 直接训练，删掉GridSearchCV
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE))
])
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
rf_scores = score_model(y_test, y_pred_rf)
st.subheader("随机森林模型得分")
st.write(rf_scores.round(2))

comparison = pd.DataFrame({
    'Baseline': baseline_scores,
    'Linear Regression': lr_scores,
    'SGD': sgd_scores,
    'Random Forest': rf_scores
}).T.round(2)
st.subheader("所有模型得分对比")
st.write(comparison)

negative_pred = (y_pred_lr < 0).sum()
st.info(f"线性回归预测负价格的手机数量：{negative_pred}")

feature_names = lr_pipeline['preprocessor'].get_feature_names_out()
weights = pd.Series(lr_pipeline['lin_reg'].coef_, index=feature_names)

cracked_diff = weights['cat__screen_condition_cracked'] - weights['cat__screen_condition_perfect']
apple_honor_diff = weights['cat__brand_Apple'] - weights['cat__brand_Honor']
repaired_cost = weights['cat__repaired_before_True']
st.write(f"碎屏相对完美屏的价格差：${cracked_diff:.0f}")
st.write(f"苹果相对荣耀的溢价：${apple_honor_diff:.0f}")
st.write(f"维修过的价格损失：${repaired_cost:.0f}")

# ========= 重点：直接删掉 permutation_importance！这个是第二大内存杀手 =========
st.warning("已跳过排列重要性计算（内存占用极高）")

def trade_in_profit(predicted, actual, offer_rate):
    offer = offer_rate * predicted
    accepted = offer >= 0.75 * actual
    profit = (actual - offer)[accepted]
    return accepted.sum(), profit.sum()

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
st.subheader("不同报价比例下的收购量和总利润")
st.write(profit_df.round(0))

X_test_age = X_test.copy()
X_test_age['phone_age'] = 2025 - X_test_age['release_year']
X_test_age['actual'] = y_test.values
X_test_age['pred_lr'] = y_pred_lr
offer_rate = 0.75
offer = offer_rate * y_pred_lr
accepted = offer >= 0.75 * y_test.values
X_test_age['profit'] = np.where(accepted, y_test.values - offer, 0)
profit_by_age = X_test_age.groupby('phone_age')['profit'].sum()
st.subheader("线性回归 0.75 报价下各机龄总利润")
st.write(profit_by_age.round(0))

gc.collect()
