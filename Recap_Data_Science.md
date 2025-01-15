# Data Science and Engineering

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Copying the data_full
data_full = pd.DataFrame()  # Placeholder for actual DataFrame
subset = data_full.copy()

# Specify the columns to drop
columns_todrop = ['DateTime', 'Date', 'DateTime_DST', 'city_name', 
                  'weather_main', 'weather_description', 'weather_icon']

# Drop the specified columns from the DataFrame
subset = subset.drop(columns=columns_todrop, axis=1)

# Generate the heatmap with increased size
plt.figure(figsize=(12, 12))

# Plot the correlation
mask = np.triu(np.ones_like(subset.corr()))
sns.heatmap(subset.corr(), mask=mask, annot=True, fmt='.2f', cmap='YlGn_r')
plt.show()
```

We will remove the following variables as their percentage is greater than 80%:

- All of the DST timestamp as we have a timestamp with date + hour
- Week as we have a date column
- Month as we have a date column
- DNI and DHI as both are used to calculate GHI
- Weather ID as we have the weather details
- Feels like as we have the exact temperature
- Temp min

```python
# Create a copy of the original dataframe 
final_df = data_full.copy()

# Remove the specified columns
columns_to_remove = ['DateTime_DST', 'Hour_DST', 'Hour_actual', 'week', 
                     'Date', 'dni', 'dhi', 'temp_min', 'temp_max',
                     'feels_like', 'weather_id',
                     'city_name', 'lat', 'lon']

final_df.drop(columns=columns_to_remove, inplace=True)

# Changing the index
final_df.set_index('DateTime', inplace=True)

# Reorder columns alphabetically
final_df = final_df.reindex(sorted(final_df.columns), axis=1)

# Move 'ghi' column to the first position
columns = final_df.columns.tolist()
columns.insert(0, columns.pop(columns.index('Actual')))
final_df = final_df[columns]

# Print the resulting dataframe
print(final_df)
```

```python
# Identify categorical and numerical columns
CAT = final_df.select_dtypes(include=['object']).columns.tolist()
NUM = final_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Print the categorical and numerical columns
print("Categorical Columns:", CAT)
print("Numerical Columns:", NUM)

# Select only the numerical columns
num_df = final_df.select_dtypes(include=['int64', 'float64']).reset_index()

# Plotting numerical columns
plt.figure(figsize=(12, 12))
for i in range(1, len(num_df.columns)):
    plt.subplot(4, 4, i)
    sns.kdeplot(x=num_df[num_df.columns[i]], label='train', color='forestgreen')
    plt.tight_layout()

plt.show()

# Plotting categorical columns
plt.figure(figsize=(15, 12))

for i in range(0, len(CAT[:-1])):
    plt.subplot(3, 4, i+1)
    ax = sns.countplot(x=final_df[CAT[i]], color='forestgreen')
    plt.xticks(rotation=45)
    plt.tight_layout()

plt.show()

# Generate a heatmap for numerical column correlations
plt.figure(figsize=(12, 12))
mask = np.triu(np.ones_like(final_df[NUM].corr()))
sns.heatmap(final_df[NUM].corr(), mask=mask, annot=True, fmt='.2f', cmap='YlGn_r')
plt.show()
```

# Data Processing

```python
# Import necessary libraries for data processing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Combine numeric and categorical features
FEATURES = NUM + CAT

# Fill missing values with mean for numeric variables
imputer = SimpleImputer(strategy='mean')
numeric_df = pd.DataFrame(imputer.fit_transform(final_df[NUM]), columns=NUM)

# Scale numeric variables using min-max scaling
scaler = MinMaxScaler()
scaled_numeric_df = pd.DataFrame(scaler.fit_transform(numeric_df), columns=NUM)

# Encode categorical variables using one-hot encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cat_df = pd.DataFrame(encoder.fit_transform(final_df[CAT]), columns=encoder.get_feature_names_out(CAT))

# Concatenate the scaled numeric and encoded categorical variables
processed_df = pd.concat([scaled_numeric_df, encoded_cat_df], axis=1)

# Display the processed dataframe
processed_df.head(5)
processed_df.shape
```

# Data Modeling

```python
# Import necessary library for splitting the dataset
from sklearn.model_selection import train_test_split

# Copying the processed dataframe
df = processed_df.copy()

# Separate features and target variable
X = df.drop('Actual', axis=1)
y = df['Actual'].copy()

# Split dataset into train and test
#X_train, X_test , y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)
X_train = X.copy()[:-2930]
y_train = y.copy()[:-2930]
X_test = X.copy()[-2930:]
y_test = y.copy()[-2930:]

# Check the shapes of the training and test data
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

# Display the training data
X_train
```

# Multiple Linear Regression

```python
# Import necessary libraries for regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# Define the final pipeline that includes the column transformer and a MLR regression model
pipe = Pipeline(steps=[('classifier', LinearRegression())])

# Fit the pipeline to the training data
pipe.fit(X_train, y_train)

# Evaluate the pipeline on the test data
score = pipe.score(X_test, y_test)
print(f'Test score: {score:.2f}')

# Evaluate the pipeline on the testing data
y_pred = pipe.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean squared error: {mse:.2f}')

# Model complexity
neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []

# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    lr = LinearRegression()
    # Fit with knn
    lr.fit(X_train, y_train)
    # Train accuracy
    train_accuracy.append(lr.score(X_train, y_train))
    # Test accuracy
    test_accuracy.append(lr.score(X_test, y_test))

# Plot
plt.plot(neig, test_accuracy, label='Testing Accuracy')
plt.plot(neig, train_accuracy, label='Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
print("Best accuracy is {}".format(np.max(test_accuracy)))

# Cross-validation
reg = LinearRegression()
k = 10
cv_result = cross_val_score(reg, X_train, y_train, cv=k)  # uses R^2 as score 
print('CV Scores: ', cv_result)
print('CV scores average: ', np.sum(cv_result)/k)
```

# XGBoost with Optuna

```python
# Import necessary libraries for XGBoost and Optuna
import xgboost as xgb
import random
import math
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import optuna
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# Set the seed for reproducibility
seed_value = 123
random.seed(seed_value)

# Train the initial XGBoost model
my_model = xgb.XGBRegressor()
my_model.fit(X_train, y_train, verbose=False)

# Make predictions
y_pred = my_model.predict(X_test)

# Calculations
print("MAE  : " + str(mean_absolute_error(y_pred, y_test)))
print("RMSE : " + str(mean_squared_error(y_test, y_pred, squared=False)))

xgb.plot_importance(my_model)

# Define the objective function for Optuna

def objective(trial):
    # Define the search space for hyperparameters 
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': trial.suggest_float('eta', 0.01, 0.3),
        'early_stopping_rounds': 100000,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 10.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0),
        'lambda': trial.suggest_float('lambda', 0.1, 10.0),
        'alpha': trial.suggest_float('alpha', 0.0, 10.0),
    }
    
    train_data, valid_data, train_target, valid_target = train_test_split(X_train, y_train, test_size=0.22, random_state=42)
    dtrain = xgb.DMatrix(train_data, label=train_target)
    dvalid = xgb.DMatrix(valid_data, label=valid_target)
    
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation-rmse')
    model = xgb.train(param, dtrain, evals=[(dvalid, 'validation')], callbacks=[pruning_callback])
    
    dtest = xgb.DMatrix(valid_data)
    y_pred = model.predict(dtest)
    rmse = mean_squared_error(valid_target, y_pred, squared=False)
    return rmse

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed_value))
study.optimize(objective, n_trials=100)

best_params = study.best_params
best_rmse = study.best_value
print("Best Hyperparameters: ", best_params)
print("Best RMSE: ", best_rmse)

# Cross-validation
random.seed(seed_value)
preds = np.zeros(X_test.shape[0])
kf = KFold(n_splits=5, random_state=48, shuffle=True)
rmse = []
n = 0

for trn_idx, test_idx in kf.split(X_train, y_train):
    X_tr, X_val = X_train, X_test
    y_tr, y_val = y_train, y_test

    model = xgb.XGBRegressor(objective="reg:linear", seed=123, **best_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    
    preds += model.predict(X_test) / kf.n_splits
    rmse.append(mean_squared_error(y_val, model.predict(X_val), squared=False))
    
    print(f"fold: {n+1} ==> rmse: {rmse[n]}")
    n += 1

# Evaluate and visualize results
scaler = MinMaxScaler()
df_train_scaled_target = scaler.fit_transform(final_df[['Actual']])

# Reshape y_pred to match the expected input shape
y_pred = y_pred.reshape(-1, 1)
predictions = scaler.inverse_transform(y_pred)
predictions = pd.DataFrame(predictions)
predictions.rename(columns={0: "Actual"}, inplace=True)

actuals = scaler.inverse_transform(y_test.reshape(-1, 1))
actuals = pd.DataFrame(actuals)
actuals.rename(columns={0: "Actual_True"}, inplace=True)
actuals[actuals < 0] = 0
predictions[predictions < 0] = 0

fig = px.scatter(x=actuals['Actual_True'], y=predictions['Actual'])
fig.show()
```

# Summary Table

```python
# Create a summary table
rank_table = {
    'Model': ['Linear Regression (GHI)', 'Linear Regression (GHI, DNI)', 
              'Random Forest', 'Two Random Forests', 'ARIMA', 
              'Autoregressive', 'Shifted Autoregressive', 'XGBoost'],
    'Accuracy Metric': [acc_LR, acc_LR2, acc_RF, 0.315, 0, 0.726, acc_SAR, xgb_acc],
    'Normalized RMSE': [RMSE_LR/max_, RMSE_LR2/max_, RMSE_RF/max_, 2492/max_, 9999, 1900/max_, RMSE_SAR/max_, xgb_score_rmse],
    'R2': [r2_LR, r2_LR2, r2_RF, 0.716, 0, 0.761, r2_SAR, xgb_r2_scores],
    'MAE': [MAE_LR, MAE_LR2, MAE_RF, 2195, 9999, 1189, MAE_SAR, (xgb_score_mae * max_)]
}

# Assign rankings based on RMSE scores
rank_table = pd.DataFrame(rank_table)
rank_table['RMSE_score'] = rank_table['Normalized RMSE'].rank(ascending=True)

# Assign rankings based on R2 scores
rank_table['R2_score'] = rank_table['R2'].rank(ascending=False)

# Assign final rank
rank_table['score'] = (0.8 * rank_table['RMSE_score']) + (0.2 * rank_table['R2_score'])
rank_table['Final_Rank'] = rank_table['score'].rank(ascending=True)

# Reorder the rows based on Final_Rank
rank_table = rank_table.sort_values(by='Final_Rank').reset_index(drop=True)

# Display table
rank_table
```
