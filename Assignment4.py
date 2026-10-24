import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

import streamlit as st
import gc

st.set_page_config(page_title="Phone Trade In Assignment 4", layout="wide")
plt.rcParams["figure.figsize"] = (8, 4)

@st.cache_data
def load_data():
    df = pd.read_csv("phone_trade_in.csv")
    return df

RANDOM_STATE = 409
df = load_data()

# ---------------------- Part 2A Data Prepare ----------------------
X = df.drop(["phone_id", "resale_price"], axis=1)
y = df["resale_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

num_features = ["release_year", "days_since_launch", "launch_price", "storage_gb", "battery_health"]
cat_features = ["brand", "tier", "screen_condition", "has_box", "repaired_before"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# -------- Question: Missing battery health analysis --------
st.header("Part 2A Data Exploration")
st.subheader("Battery health missing share by brand (train set)")
battery_missing = X_train.groupby("brand")["battery_health"].apply(lambda s: s.isna().mean())
st.write(battery_missing.round(3))

X_train_age = X_train.copy()
X_train_age["phone_age"] = 2025 - X_train_age["release_year"]

st.subheader("Battery health median by brand")
batt_med_brand = X_train.groupby("brand")["battery_health"].median()
st.write(batt_med_brand.round(1))

st.subheader("Battery health median by phone age")
batt_med_age = X_train_age.groupby("phone_age")["battery_health"].median()
st.write(batt_med_age.round(1))

st.subheader("Resale price vs days_since_launch scatter plot")
fig, ax = plt.subplots()
ax.scatter(X_train["days_since_launch"], y_train, alpha=0.3, s=8)
ax.set_xlabel("days_since_launch")
ax.set_ylabel("resale_price")
ax.set_title("Training set resale_price vs days_since_launch")
st.pyplot(fig)

# ---------------------- Part 2B Train Models ----------------------
st.header("Part 2B Model Training & Comparison")

# Step1: Baseline (DummyRegressor mean)
baseline_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("dummy", DummyRegressor(strategy="mean"))
])
baseline_pipeline.fit(X_train, y_train)
y_pred_baseline = baseline_pipeline.predict(X_test)

# Step2: Linear Regression
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("lin_reg", LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

# Step3: SGDRegressor GridSearchCV
sgd_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("sgd_reg", SGDRegressor(random_state=RANDOM_STATE))
])
sgd_param_grid = {
    "sgd_reg__max_iter": [10000, 100000],
    "sgd_reg__tol": [1e-5, 1e-3],
    "sgd_reg__eta0": [0.01, 0.1]
}
sgd_grid = GridSearchCV(
    sgd_pipeline,
    param_grid=sgd_param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)
sgd_grid.fit(X_train, y_train)
sgd_best = sgd_grid.best_estimator_
y_pred_sgd = sgd_best.predict(X_test)
st.subheader("SGD Best Params")
st.write(sgd_grid.best_params_)

# Step4: RandomForest GridSearchCV
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("rf", RandomForestRegressor(random_state=RANDOM_STATE))
])
rf_param_grid = {
    "rf__n_estimators": [100, 150],
    "rf__max_depth": [5, 10]
}
rf_grid = GridSearchCV(
    rf_pipeline,
    param_grid=rf_param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)
st.subheader("Random Forest Best Params")
st.write(rf_grid.best_params_)

# Score function
def get_scores(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return pd.Series({"RMSE": rmse, "MAE": mae, "R2": r2})

score_baseline = get_scores(y_test, y_pred_baseline)
score_lr = get_scores(y_test, y_pred_lr)
score_sgd = get_scores(y_test, y_pred_sgd)
score_rf = get_scores(y_test, y_pred_rf)

score_table = pd.DataFrame({
    "Baseline": score_baseline,
    "Linear Regression": score_lr,
    "SGD": score_sgd,
    "Random Forest": score_rf
}).T.round(2)

st.subheader("Model Test Set Scores Table")
st.write(score_table)

# Count negative predicted price for linear regression
neg_count_lr = (y_pred_lr < 0).sum()
st.write(f"Linear regression predicted negative resale price on {neg_count_lr} test phones.")

# ---------------------- Part 2C Interpret ----------------------
st.header("Part 2C Model Interpretation")
st.subheader("Linear Regression Coefficients")
weights = pd.Series(
    lr_pipeline["lin_reg"].coef_,
    index=lr_pipeline["preprocessor"].get_feature_names_out()
)
st.write(weights.round(0))

st.subheader("Permutation Importance | Linear Regression (Top5)")
perm_lr = permutation_importance(
    lr_pipeline, X_test, y_test,
    scoring="neg_root_mean_squared_error",
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=1
)
perm_lr_df = pd.Series(perm_lr.importances_mean, index=X_test.columns).sort_values(ascending=False).head(5)
st.write(perm_lr_df.round(2))

st.subheader("Permutation Importance | Best Random Forest (Top5)")
perm_rf = permutation_importance(
    rf_best, X_test, y_test,
    scoring="neg_root_mean_squared_error",
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=1
)
perm_rf_df = pd.Series(perm_rf.importances_mean, index=X_test.columns).sort_values(ascending=False).head(5)
st.write(perm_rf_df.round(2))

# ---------------------- Part 2D Trade-in Profit Decision ----------------------
st.header("Part 2D Trade-in Profit Calculation")

def trade_in_profit(predicted, actual, offer_rate):
    offer = offer_rate * predicted
    accepted = offer >= 0.75 * actual
    profit = (actual - offer)[accepted]
    return accepted.sum(), profit.sum()

offer_rates = np.arange(0.60, 0.91, 0.05)
profit_rows = []
for r in offer_rates:
    n_base, p_base = trade_in_profit(y_pred_baseline, y_test.to_numpy(), r)
    n_lr, p_lr = trade_in_profit(y_pred_lr, y_test.to_numpy(), r)
    n_rf, p_rf = trade_in_profit(y_pred_rf, y_test.to_numpy(), r)
    profit_rows.append({
        "offer_rate": round(r,2),
        "baseline_phones": n_base,
        "baseline_profit": p_base,
        "lr_phones": n_lr,
        "lr_profit": p_lr,
        "rf_phones": n_rf,
        "rf_profit": p_rf
    })
profit_df = pd.DataFrame(profit_rows).set_index("offer_rate")
st.subheader("Profit and accepted count at different offer rates")
st.write(profit_df.round(0))

# LR 0.75 offer rate profit by phone age
X_test_age = X_test.copy()
X_test_age["phone_age"] = 2025 - X_test_age["release_year"]
X_test_age["actual"] = y_test.values
X_test_age["pred_lr"] = y_pred_lr
offer_75 = 0.75 * X_test_age["pred_lr"]
accepted_75 = offer_75 >= 0.75 * X_test_age["actual"]
X_test_age["profit_75"] = np.where(accepted_75, X_test_age["actual"] - offer_75, 0)
profit_by_age_lr75 = X_test_age.groupby("phone_age")["profit_75"].sum()
st.subheader("LR @0.75 offer rate profit grouped by phone age")
st.write(profit_by_age_lr75.round(0))

# ---------------------- Part 2E Manager Letter ----------------------
st.header("Part 2E Advice to Platform Manager")
letter = """
I recommend you use our more accurate pricing tool and set seller offers at 75% of its estimated resale value.
Running on our 1000 test phones, this setup brings a total profit of [fill your rf 0.75 profit number] dollars.
If you set offers too low, very few sellers will accept and you will miss out on buying inventory.
If you set offers too high, you pay too much for phones and lose money on each deal.
The main risk is that the tool can still misjudge some older phones and overpay for them.
"""
st.markdown(letter)

gc.collect()
