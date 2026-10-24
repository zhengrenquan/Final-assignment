from pathlib import Path
import json, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from IPython.display import display
from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

SEED = 409
OUT = Path('results'); OUT.mkdir(exist_ok=True)
FIG = Path('figures'); FIG.mkdir(exist_ok=True)
answers = {}
plt.rcParams.update({'figure.figsize': (9, 5), 'figure.dpi': 120, 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': .18, 'savefig.bbox': 'tight'})
print('Python:', platform.python_version(), '| sklearn:', sklearn.__version__, '| pandas:', pd.__version__)
df = pd.read_csv('phone_trade_in.csv')
X = df.drop(columns=['phone_id', 'resale_price'])
y = df['resale_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=409)
assert len(df) == 5000 and len(X_train) == 4000 and len(X_test) == 1000
assert set(X_train.index).isdisjoint(X_test.index)
train = X_train.assign(resale_price=y_train, age=2025-X_train.release_year)
print(f'Train: {len(X_train):,} | Test: {len(X_test):,} | Input features: {X.shape[1]}')
display(train.head())
display(train.describe(include='all').T)
display(X_train.isna().sum().rename('Training missing count').to_frame())
numeric_features = X_train.select_dtypes(include='number').columns.tolist()
categorical_features = X_train.select_dtypes(exclude='number').columns.tolist()
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                      ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
])
print('Numeric:', numeric_features)
print('Categorical:', categorical_features)

global_battery = train.battery_health.median()
battery_by_brand = train.groupby('brand').agg(
    phones=('battery_health', 'size'), observed=('battery_health', 'count'), median=('battery_health', 'median'))
battery_by_brand['missing_pct'] = 100*(1-battery_by_brand.observed/battery_by_brand.phones)
battery_by_brand['global_minus_group_median'] = global_battery-battery_by_brand['median']
battery_by_age = train.groupby('age').agg(phones=('battery_health', 'size'), median=('battery_health', 'median'))
battery_by_age['global_minus_group_median'] = global_battery-battery_by_age['median']
print('Global training battery median:', global_battery)
display(battery_by_brand.round(2))
display(battery_by_age.round(2))
battery_by_brand.to_csv(OUT/'battery_by_brand.csv')
battery_by_age.to_csv(OUT/'battery_by_age.csv')
brand_text = '; '.join(f"{b}: {r.missing_pct:.1f}% missing, median {r['median']:.1f}%" for b,r in battery_by_brand.iterrows())
age_text = '; '.join(f"age {a}: {r['median']:.1f}%" for a,r in battery_by_age.iterrows())
high_age = battery_by_age.index[battery_by_age['median']<global_battery].tolist()
low_age = battery_by_age.index[battery_by_age['median']>global_battery].tolist()
high_brand = battery_by_brand.index[battery_by_brand['median']<global_battery].tolist()
low_brand = battery_by_brand.index[battery_by_brand['median']>global_battery].tolist()
answers['q1'] = (f"### Q1 — Missing battery health\n{brand_text}. Observed battery medians by age: {age_text}. "
    f"The overall training median is **{global_battery:.1f}%**. Relative to these observed group medians, it imputes too high for ages {high_age} "
    f"and brands {high_brand}, and too low for ages {low_age} and brands {low_brand}. Equal-median groups have no group-median discrepancy. "
    "These are comparisons with observed medians, not proof of the unobserved batteries' true values. "
    "The models also observe release_year and days_since_launch, so they can partly recover age-related depreciation even when battery health is imputed. "
    "Imputation still loses individual battery information and brand-dependent missingness may matter; the median is always fitted inside each training fold.")

fig, ax = plt.subplots()
ax.scatter(train.days_since_launch, train.resale_price, s=10, alpha=.22, color='#0072B2', edgecolors='none')
ax.set(xlabel='Days since launch', ylabel='Resale price (USD)', title='Training phones: resale value over time')
fig.savefig(FIG/'price_over_time.png', dpi=180)
plt.show()
age_price = train.groupby('age').resale_price.agg(['count','median','mean'])
display(age_price.round(2))
answers['q2'] = (f"### Q2 — Price over time\nResale prices generally decrease with time on the market: the training median is "
    f"**${age_price.loc[0,'median']:.2f} at age 0**, **${age_price.loc[3,'median']:.2f} at age 3**, and "
    f"**${age_price.loc[6,'median']:.2f} at age 6**. "
    "The scatter has a broad range among new phones and a narrower low-price region among old phones, indicating curved depreciation and differences in launch price, tier and condition. "
    "One straight line is a poor description of the full pattern because the same linear slope cannot capture both early depreciation and the low-price floor.")

baseline_value = float(y_train.mean())
lr_pipeline = Pipeline([('preprocessor', clone(preprocessor)), ('lin_reg', LinearRegression())])
lr_pipeline.fit(X_train, y_train)
sgd_pipeline = Pipeline([('preprocessor', clone(preprocessor)), ('sgd', SGDRegressor(random_state=409))])
sgd_grid = {'sgd__max_iter':[10000,100000], 'sgd__tol':[1e-5,1e-3], 'sgd__eta0':[0.01,0.1]}
sgd_search = GridSearchCV(sgd_pipeline, sgd_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=2, error_score='raise')
sgd_search.fit(X_train, y_train)
print('SGD best parameters:', sgd_search.best_params_)
print('SGD best training-CV RMSE:', -sgd_search.best_score_)
rf_pipeline = Pipeline([('preprocessor', clone(preprocessor)), ('rf', RandomForestRegressor(random_state=409))])
rf_grid = {'rf__n_estimators':[100,150], 'rf__max_depth':[5,10]}
rf_search = GridSearchCV(rf_pipeline, rf_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=2, error_score='raise')
rf_search.fit(X_train, y_train)
print('RF best parameters:', rf_search.best_params_)
print('RF best training-CV RMSE:', -rf_search.best_score_)
print('Training-mean baseline: $', round(baseline_value,2))
for label, search in [('sgd',sgd_search), ('rf',rf_search)]:
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results['cv_rmse'] = -cv_results.mean_test_score
    display(cv_results[['params','cv_rmse','std_test_score','rank_test_score']].sort_values('rank_test_score'))
    cv_results.to_csv(OUT/f'{label}_cv.csv', index=False)
models = {'Linear regression':lr_pipeline, 'SGD':sgd_search.best_estimator_, 'Random forest':rf_search.best_estimator_}
predictions = {'Baseline':np.full(len(X_test), baseline_value)}
predictions.update({name:model.predict(X_test) for name,model in models.items()})
actual = y_test.to_numpy()
scores = pd.DataFrame({name:{'RMSE':np.sqrt(mean_squared_error(actual,pred)),
                           'MAE':mean_absolute_error(actual,pred), 'R_squared':r2_score(actual,pred)}
                       for name,pred in predictions.items()}).T
display(scores.round(4))
improvements = pd.DataFrame({'RMSE reduction':scores.loc['Baseline','RMSE']-scores.RMSE,
                           'MAE reduction':scores.loc['Baseline','MAE']-scores.MAE,
                           'R_squared increase':scores.R_squared-scores.loc['Baseline','R_squared']})
display(improvements.round(4))
scores.to_csv(OUT/'test_scores.csv')
pred_out = pd.DataFrame(predictions, index=X_test.index).assign(actual=y_test, phone_id=df.loc[X_test.index,'phone_id'])
pred_out.to_csv(OUT/'test_predictions.csv', index_label='original_row')
negative_counts = {k:int((v<0).sum()) for k,v in predictions.items()}
print('Negative-price counts:', negative_counts)
winner = scores.RMSE.idxmin()
answers['q3'] = (f"### Q3 — Which model is best?\n**{winner}** wins on RMSE ({scores.loc[winner,'RMSE']:.2f} USD), "
    f"MAE ({scores.loc[winner,'MAE']:.2f} USD) and R-squared ({scores.loc[winner,'R_squared']:.4f}). "
    + ' '.join(f"{name} reduces baseline RMSE by **${row['RMSE reduction']:.2f}**, reduces MAE by **${row['MAE reduction']:.2f}**, "
               f"and raises R-squared by **{row['R_squared increase']:.4f}**." for name,row in improvements.drop(index='Baseline').iterrows()))
lr, sgd = scores.loc['Linear regression'], scores.loc['SGD']
answers['q4'] = (f"### Q4 — Same model, two algorithms\nLinear regression and SGD have test RMSE **${lr.RMSE:.2f} vs ${sgd.RMSE:.2f}** "
    f"(absolute difference **${abs(lr.RMSE-sgd.RMSE):.2f}**), MAE **${lr.MAE:.2f} vs ${sgd.MAE:.2f}**, and R-squared "
    f"**{lr.R_squared:.4f} vs {sgd.R_squared:.4f}**. Both use additive linear predictions on identical transformed features, so they should have similar accuracy. "
    "Least squares solves directly whereas SGD makes approximate stochastic updates and stops according to its tolerance; SGD's default L2 penalty also changes the objective slightly. "
    "Thus their predictions need not match exactly; scaling numeric inputs helps SGD converge.")
answers['q5'] = (f"### Q5 — Why the forest wins\nLinear regression predicts negative resale prices for **{negative_counts['Linear regression']} of 1,000 test phones** "
    f"({negative_counts['Linear regression']/10:.1f}%), while the forest produces **{negative_counts['Random forest']}**. "
    f"The forest's RMSE is **${scores.loc['Random forest','RMSE']:.2f}**, compared with **${lr.RMSE:.2f}** for linear regression. "
    "Question 2 shows curved depreciation and a low-price floor; an additive straight-line model can extend below zero and cannot represent price-dependent condition effects well. "
    "Trees split phones into local groups and capture nonlinearities and interactions; their leaf averages of nonnegative training prices stay nonnegative. "
    "Nonnegativity alone does not guarantee accuracy, but the held-out score supports the forest's advantage here.")

weights = pd.Series(lr_pipeline['lin_reg'].coef_, index=lr_pipeline['preprocessor'].get_feature_names_out())
print(weights.round(0).to_string())
weights.to_csv(OUT/'linear_weights.csv', header=['coefficient'])
cracked_effect = weights['cat__screen_condition_cracked']-weights['cat__screen_condition_perfect']
apple_effect = weights['cat__brand_Apple']-weights['cat__brand_Honor']
repair_effect = weights['cat__repaired_before_yes']-weights['cat__repaired_before_no']
contrasts = pd.Series({'Cracked minus perfect':cracked_effect,'Apple minus Honor':apple_effect,'Repaired yes minus no':repair_effect})
display(contrasts.round(2).rename('Price difference (USD)').to_frame())
answers['q6'] = (f"### Q6 — What the linear model says\nHolding other encoded features fixed, cracked minus perfect is **${cracked_effect:+.2f}**, "
    f"Apple minus Honor is **${apple_effect:+.2f}**, and repaired-before yes minus no is **${repair_effect:+.2f}**. "
    "A negative difference is a predicted price reduction. These are within-feature contrasts, not the raw dummy coefficients or causal guarantees.")
answers['q7'] = (f"### Q7 — One number for every phone\nThe linear model applies the same **${abs(cracked_effect):.2f}** cracked-screen adjustment "
    f"to both a $150 and a $900 phone: its magnitude is **{abs(cracked_effect)/150*100:.1f}%** vs **{abs(cracked_effect)/900*100:.1f}%** of those prices. "
    "That fixed dollar adjustment is unlikely to hold across all tiers, ages and launch prices. A forest can split on price or tier before screen condition, "
    "allowing different screen penalties for inexpensive and expensive phones rather than imposing one global additive penalty.")

importance_tables = {}
for name,model in [('Linear regression',lr_pipeline),('Random forest',rf_search.best_estimator_)]:
    imp = permutation_importance(model, X_test, y_test, scoring='neg_root_mean_squared_error', n_repeats=10, random_state=409, n_jobs=2)
    table = pd.DataFrame({'feature':X_test.columns, 'RMSE increase USD':imp.importances_mean,
                          'repeat SD USD':imp.importances_std}).sort_values('RMSE increase USD',ascending=False).reset_index(drop=True)
    importance_tables[name]=table
    print(name, '— five most important raw features')
    display(table.head(5).round(2))
    table.to_csv(OUT/('importance_'+name.replace(' ','_')+'.csv'), index=False)
fig, axes = plt.subplots(1,2,figsize=(12,4.5), layout='constrained')
for ax,(name,table) in zip(axes,importance_tables.items()):
    top=table.head(5).iloc[::-1]
    ax.barh(top.feature,top['RMSE increase USD'],xerr=top['repeat SD USD'],color='#0072B2',capsize=3)
    ax.set(title=name,xlabel='Increase in RMSE (USD)')
fig.savefig(FIG/'permutation_importance.png',dpi=180)
plt.show()
shared = sorted(set(importance_tables['Linear regression'].head(5).feature)&set(importance_tables['Random forest'].head(5).feature))
answers['q8'] = ("### Q8 — What drives the price?\n" + ' '.join(
    name + ': ' + '; '.join(f"{r['feature']} **+${r['RMSE increase USD']:.2f}**" for _,r in table.head(5).iterrows()) + '.'
    for name,table in importance_tables.items()) + f" Their top-five overlap is **{shared}**. "
    "Larger increases mean shuffling that feature damages prediction more. Error bars are the standard deviation across 10 shuffles, not confidence intervals. "
    "Correlated variables such as release_year and days_since_launch share information, so their separate permutation rankings should not be read as causal rankings.")

def trade_in_profit(predicted, actual, offer_rate):
    """Offer offer_rate x predicted price; a seller accepts if the offer is at least
    75% of what the phone is really worth; the platform resells at that worth."""
    offer = offer_rate * predicted
    accepted = offer >= 0.75 * actual
    profit = (actual - offer)[accepted]
    return accepted.sum(), profit.sum()

rates = np.round(np.arange(.60,.901,.05),2)
decision_models = ['Baseline','Linear regression','Random forest']
rows=[]
for rate in rates:
    row={'offer_rate':rate}
    for name in decision_models:
        bought,profit = trade_in_profit(predictions[name],actual,rate)
        row[name+' bought']=int(bought)
        row[name+' profit']=float(profit)
    rows.append(row)
profit_table = pd.DataFrame(rows).set_index('offer_rate')
display(profit_table.round(2))
profit_table.to_csv(OUT/'offer_scenarios.csv')
fig,ax=plt.subplots()
for name,color,marker in zip(decision_models,['#777777','#D55E00','#0072B2'],['s','^','o']):
    ax.plot(rates,profit_table[name+' profit'],label=name,color=color,marker=marker)
ax.axhline(0,color='black',linewidth=.8)
ax.set(xlabel='Offer rate',ylabel='Total profit (USD)',title='Same 1,000 test phones: offer-rate scenarios')
ax.legend()
fig.savefig(FIG/'offer_profit.png',dpi=180)
plt.show()
best_rate=float(profit_table['Random forest profit'].idxmax())
best_profit=float(profit_table.loc[best_rate,'Random forest profit'])
best_bought=int(profit_table.loc[best_rate,'Random forest bought'])
answers['q9'] = (f"### Q9 — The best offer\nThe forest's highest total profit among the seven tested rates is **${best_profit:,.2f}** "
    f"at **{best_rate:.0%}**, buying **{best_bought} of 1,000** phones. At 60% it buys **{int(profit_table.loc[.6,'Random forest bought'])}** "
    f"and earns **${profit_table.loc[.6,'Random forest profit']:,.2f}**; at 90% it buys **{int(profit_table.loc[.9,'Random forest bought'])}** "
    f"and earns **${profit_table.loc[.9,'Random forest profit']:,.2f}**. Low offers fail the 75%-of-actual-value acceptance threshold and lose sellers; "
    "high offers reduce margins and can overpay for individual phones. This optimum is retrospective on this sample, not an independently validated future policy.")

baseline_rows=[]
for rate in rates:
    offer=rate*baseline_value
    accepted=offer>=.75*actual
    bought_values=actual[accepted]
    baseline_rows.append({'offer_rate':rate,'flat_offer':offer,'acceptance_value_ceiling':offer/.75,
                         'bought':int(accepted.sum()),'mean_actual_accepted':bought_values.mean(),
                         'accepted_age_mean':(2025-X_test.loc[accepted,'release_year']).mean(),
                         'loss_making_phones':int((bought_values<offer).sum()),
                         'profit':float((bought_values-offer).sum())})
baseline_diagnostics=pd.DataFrame(baseline_rows).set_index('offer_rate')
display(baseline_diagnostics.round(2))
baseline_diagnostics.to_csv(OUT/'baseline_diagnostics.csv')
b=baseline_diagnostics.loc[.75]
answers['q10'] = (f"### Q10 — The baseline\nThe training mean is **${baseline_value:.2f}**, so at 75% the platform offers every seller "
    f"**${b.flat_offer:.2f}**. Only phones worth at most **${b.acceptance_value_ceiling:.2f}** accept; it buys **{int(b.bought)}**, "
    f"whose average actual resale value is only **${b.mean_actual_accepted:.2f}**, producing **${b.profit:,.2f}** profit. "
    f"Across rates, total profits range from **${baseline_diagnostics.profit.min():,.2f}** to **${baseline_diagnostics.profit.max():,.2f}**. "
    "The fixed offer attracts low-value phones while more valuable phones leave: adverse selection makes average resale revenue lower than the payment here. "
    "Strictly, acceptance alone does not mathematically guarantee a loss on every phone: phones with actual values between the offer and offer/0.75 still generate nonnegative margins. "
    "The table establishes an aggregate loss at every tested rate for this dataset, rather than a universal guarantee.")

lr_offer=.75*predictions['Linear regression']
lr_accepted=lr_offer>=.75*actual
lr_phone_profit=np.where(lr_accepted,actual-lr_offer,0.0)
age_diagnostics=pd.DataFrame({'age':2025-X_test.release_year.to_numpy(),'actual':actual,
    'predicted':predictions['Linear regression'],'bought':lr_accepted.astype(int),'profit':lr_phone_profit,
    'paid':np.where(lr_accepted,lr_offer,0.0),'resale_revenue':np.where(lr_accepted,actual,0.0)})
by_age=age_diagnostics.groupby('age').agg(phones=('profit','size'),bought=('bought','sum'),
    total_profit=('profit','sum'),profit_per_phone=('profit','mean'),mean_actual=('actual','mean'),
    mean_predicted=('predicted','mean'),paid=('paid','sum'),resale_revenue=('resale_revenue','sum'))
assert np.isclose(by_age.total_profit.sum(),profit_table.loc[.75,'Linear regression profit'])
display(by_age.round(2))
by_age.to_csv(OUT/'linear_profit_by_age.csv')
fig,ax=plt.subplots()
ax.bar(by_age.index,by_age.total_profit,color=['#0072B2' if x>=0 else '#D55E00' for x in by_age.total_profit])
ax.axhline(0,color='black',linewidth=.8)
ax.set(xlabel='Phone age (2025 - release year)',ylabel='Total profit (USD)',title='Linear model at 75%: profit by age, non-purchases = 0')
fig.savefig(FIG/'linear_profit_by_age.png',dpi=180)
plt.show()
loss_ages=by_age[by_age.total_profit<0]
answers['q11'] = (f"### Q11 — R-squared is not profit\nLinear regression has test R-squared **{scores.loc['Linear regression','R_squared']:.4f}**, "
    f"yet its 75% offer produces **${by_age.total_profit.sum():,.2f}** total profit. Loss-making age groups: "
    + '; '.join(f"age {age}: **${row.total_profit:,.2f}** from {int(row.bought)}/{int(row.phones)} purchases" for age,row in loss_ages.iterrows())
    + f". These groups lose **${loss_ages.total_profit.sum():,.2f}** combined. "
    "The zero-inclusive age table shows where overpayment accumulates: predicted and actual values diverge, and sellers accept generous offers on overvalued phones. "
    "Underpriced phones reject offers and cannot offset the losses. R-squared summarizes symmetric prediction errors on all phones; profit depends asymmetrically on accepted purchases, "
    "so a respectable global fit does not imply a profitable acquisition policy.")

manager_numbers = pd.Series({'recommended_offer_rate':best_rate,'phones_evaluated':1000,
    'phones_bought':best_bought,'gross_profit_USD':best_profit,
    'baseline_profit_at_same_rate':profit_table.loc[best_rate,'Baseline profit'],
    'linear_profit_at_same_rate':profit_table.loc[best_rate,'Linear regression profit']})
display(manager_numbers.to_frame('Value').round(2))
answers['q12'] = (f"### Q12 — Message to the platform manager\n"
    f"Use the pricing system that compares phones within detailed groups by their launch price, age and condition, and initially offer sellers **{best_rate:.0%}** of its predicted resale value. "
    f"In our simulation of **1,000 phones**, this approach purchased **{best_bought}** and generated **${best_profit:,.2f}** before operating costs. "
    f"At the same offer percentage, the flat-price approach returned **${profit_table.loc[best_rate,'Baseline profit']:,.2f}** and the simpler price formula returned **${profit_table.loc[best_rate,'Linear regression profit']:,.2f}**. "
    "Treat this amount as a planning benchmark rather than a revenue promise, because these are simulated records and the offer percentage was selected after comparing this sample. "
    "Before launch, run a small independent pilot and account for inspection, repairs, shipping, fraud and differences in seller willingness to accept an offer. "
    "Review purchase margins regularly and update the pricing system when demand or phone values change.")

assert len(answers)==12
assert set(answers)=={f'q{i}' for i in range(1,13)}
assert np.isfinite(scores.to_numpy()).all()
assert profit_table.shape==(7,6)
assert len(pred_out)==1000
for name in decision_models:
    for rate in rates:
        assert 0 <= profit_table.loc[rate,name+' bought'] <= 1000
Path('results/answers.json').write_text(json.dumps(answers,ensure_ascii=False,indent=2),encoding='utf-8')
print('Audit passed: 12 answers, 4 model evaluations, 7 offer scenarios, 1,000 test predictions.')