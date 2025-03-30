import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import sys
import os
sys.path.append(os.path.expanduser("~/CO2-to-C3/src"))
from ml_tools import filter_max_jtarget, create_pb_feature_df
from utils.help_utils import get_data_df

from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV
from scipy.stats import uniform, randint
from xgboost import XGBRegressor
import shap

import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

# Import Data
data_path = "~/data"    # Directory path where you stored the data – Dataset available at https://doi.org/10.5281/zenodo.15107045

pb_df = get_data_df(data_path, "Pourbaix_phase.xlsx")
ocp_df = get_data_df(data_path, "OCP_Eads.xlsx")
co2r_df = get_data_df(data_path, "CO2R_dataset.xlsx")

# Set parameters
category = 'all'                # 'Cu', 'nonCu', 'all'
target = 'C3H6'                 # 'C3H6', 'C3H8'
batch = [0,1,2,3,4,5,6]         # Batch 0 means Seed dataset 
millers = [(1,0,0),(2,1,1)]     
species_columns = ['*OCHO','*COOH','CO*COH-2*CO','*CHO-*CO','*C-*CHO']

random_seed = 7       

# Preprocess data
co2r_df = co2r_df[co2r_df['batch'].isin(batch)]

if category == 'all':
    pass
elif category == 'Cu':
    co2r_df = co2r_df[co2r_df['composition'].str.contains('Cu')]
elif category == 'nonCu':
    co2r_df = co2r_df[~(co2r_df['composition'].str.contains('Cu'))]
else:
    raise ValueError("Invalid category")

co2r_df[f'j_{target}'] = co2r_df['current_density'] * co2r_df['FE_' + target + '_mean'] / 100
co2r_df = filter_max_jtarget(co2r_df, target)

print("Total number of data: ", len(co2r_df))


# Create feature DataFrame
feature_df = create_pb_feature_df(co2r_df, ocp_df, pb_df, millers = millers)

# Create Train DataFrame
train_df = pd.merge(feature_df, co2r_df, on='composition')
train_df[f'j_{target}_log10'] = train_df[f'j_{target}'].apply(lambda x: np.nan if x == 0 else np.log10(x))
train_df = train_df[~train_df[f'j_{target}_log10'].isna()]

# Set Train/Target col
train_cols = ['f1', 'f2', 'f3'] +\
            ['ele1_' + x for x in species_columns] +\
            ['ele2_' + x for x in species_columns] +\
            ['ele3_' + x for x in species_columns] +\
            ['pb1_' + x for x in species_columns] +\
            ['pb2_' + x for x in species_columns]

X = train_df[train_cols].copy()
y = train_df[f'j_{target}_log10'].copy()

# Hyperparameter Tuning
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(2, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 7),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}

base_model = XGBRegressor(
    missing=np.nan,
    random_state=random_seed
)

print("Starting hyperparameter optimization...")

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=200,  # Number of parameter settings sampled
    cv=5,        # 5-fold cross validation
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,   # Use all available cores
    verbose=2,
    random_state=random_seed
)

random_search.fit(X, y)

print("\nBest parameters found:")
for param, value in random_search.best_params_.items():
    print(f"{param}: {value}")

print(f"\nBest CV score: {-random_search.best_score_:.4f} RMSE")
print()

# LOOCV
best_model = random_search.best_estimator_
loo = LeaveOneOut()
y_true = []
y_pred = []
all_shap_values = []

print("\nStarting LOOCV with best model...")

for train_idx, test_idx in loo.split(X):
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train model
    best_model.fit(X_train, y_train)
    
    # Make prediction
    pred = best_model.predict(X_test)
    
    y_true.append(y_test.iloc[0])
    y_pred.append(pred[0])

    # SHAP values
    explainer = shap.TreeExplainer(best_model)
    all_shap_values.append(explainer.shap_values(X_test))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

r2 = np.corrcoef(y_true, y_pred)[0,1]**2
rmse = np.sqrt(np.mean((y_true - y_pred)**2))
pearsonr = np.corrcoef(y_true, y_pred)[0,1]

print("\nFinal Results with Best Model:")
print(f"Overall R² Score: {r2:.4f}")
print(f"Overall RMSE: {rmse:.4f}")

# Create LOOCV plot
plt.figure(figsize=(10, 10))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title(f'LOOCV Predictions (Best Model)\nR² = {r2:.3f}, RMSE = {rmse:.3f}, PearsonR = {pearsonr:.3f}')
plt.grid(True)

plt.show()

plt.close()


# SHAP Analysis

# SHAP Values - grouped by Eads features
transformed_shap_values = []
n_features = len(species_columns)

for i in range(len(all_shap_values)):

    current_row = all_shap_values[i][0]

    ratios = current_row[:3]

    reshaped_values = current_row[3:].reshape(5, n_features)

    summed_values = np.nansum(reshaped_values, axis=0) 

    combined_values = np.concatenate([ratios, summed_values])

    transformed_shap_values.append(combined_values)

transformed_shap_values = np.array(transformed_shap_values)

# X values - grouped by Eads features
X = train_df[train_cols].copy()
X_array = X.to_numpy()

X_transformed = []

for i in range(X_array.shape[0]):
    
    current_row = X_array[i]

    ratios = current_row[:3]

    reshaped_values = current_row[3:].reshape(5, n_features)

    summed_values = np.nanmean(reshaped_values, axis=0)  

    combined_values = np.concatenate([ratios, summed_values])  
    
    X_transformed.append(combined_values)

X_transformed = np.array(X_transformed)
X_transformed_df = pd.DataFrame(X_transformed, columns=['f1','f2','f3'] + species_columns)

# Filter columns and SHAP values
filtered_features = X_transformed_df[species_columns]  
filtered_shap_values = transformed_shap_values[:, 3:]  # Remove ratios

# Create SHAP summary plot
shap.summary_plot(filtered_shap_values, 
                  features=filtered_features, 
                  feature_names=species_columns, 
                  plot_size=[6,5], show=False, plot_type='dot')

fig = plt.gcf()
fig.set_dpi(360)

plt.gca().spines['bottom'].set_linewidth(1.5)
plt.axvline(x=0, color='black', alpha=1, linewidth=1.5)

plt.xticks(fontsize=12, color='black')
plt.tick_params(axis='x', which='major', width=1.5, length=5, color='black')
plt.xlabel('SHAP value', fontsize=12, fontdict={'family': 'sans-serif', 'size': 12, })

plt.yticks(fontsize=12)
plt.gca().set_yticklabels(
    [label.get_text() for label in plt.gca().get_yticklabels()],
    fontdict={'family': 'sans-serif', 'size': 14, 'color': 'black'}
)

plt.show()