## 1. Predict Citizen Income Slab

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Step 2: Data Preprocessing
## Convert 'age' column to numeric (handling cases where it's written in words)
def convert_age(age):
    try:
        return int(age)
    except ValueError:
        return np.nan

train_df["age"] = train_df["age"].apply(convert_age)
test_df["age"] = test_df["age"].apply(convert_age)

## Extract numeric part from 'fin_wt_working_hours'
def extract_hours(value):
    try:
        return float(str(value).split(";")[0].split(",")[0].split("_")[0])
    except:
        return np.nan

train_df["fin_wt_working_hours"] = train_df["fin_wt_working_hours"].apply(extract_hours)
test_df["fin_wt_working_hours"] = test_df["fin_wt_working_hours"].apply(extract_hours)

## Fill missing values with median or mode
train_df.fillna(train_df.median(numeric_only=True), inplace=True)
test_df.fillna(test_df.median(numeric_only=True), inplace=True)

## Encode categorical variables
categorical_columns = ["workclass", "education", "marital_status", "relationship_to_dependent", "ethnicity", "gender"]
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

## Standardize numerical features
scaler = StandardScaler()
num_cols = ["age", "fin_wt_working_hours", "years_of_study", "capital_earned", "capital_spent"]
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

# Step 3: Data Visualization
## Histograms of numerical features
train_df.hist(figsize=(12, 10))
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

## Categorical Feature Distribution
for col in categorical_columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x=train_df[col])
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.show()

# Step 4: Define features and target variable
X = train_df.drop(columns=["id", "outcome"])
y = train_df["outcome"]

# Step 5: Split train-validation for model selection
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Feature Importance Visualization
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh', figsize=(10,6))
plt.title("Top 10 Feature Importances")
plt.show()

# Step 8: Predict and evaluate on validation set
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Step 9: Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Low", "High"], yticklabels=["Low", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 10: Predict on test data
X_test = test_df.drop(columns=["id"])
test_predictions = model.predict(X_test)

# Step 11: Save predictions in required format
submission_df = pd.DataFrame({"id": test_df["id"], "outcome": test_predictions})
submission_df.to_csv("submissions.csv", index=False)

print("Submission file saved as submissions.csv")

```

---

## 2 . Predict Flight Booking

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

# Step 1: Load datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Step 2: Data Preprocessing
## Encode categorical variables
categorical_columns = ["Gender", "Class", "Inflight wifi service", "Food and drink"]
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

## Convert 'Departure Delay in Minutes' to numeric (handling negative values)
train_df["Departure Delay in Minutes"] = pd.to_numeric(train_df["Departure Delay in Minutes"], errors='coerce')
test_df["Departure Delay in Minutes"] = pd.to_numeric(test_df["Departure Delay in Minutes"], errors='coerce')

## Fill missing values with median
train_df.fillna(train_df.median(numeric_only=True), inplace=True)
test_df.fillna(test_df.median(numeric_only=True), inplace=True)

## Standardize numerical features
scaler = StandardScaler()
num_cols = ["Age", "Flight Distance", "Departure Delay in Minutes"]
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

# Step 3: Data Visualization
## Histograms of numerical features
train_df.hist(figsize=(12, 10))
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

## Categorical Feature Distribution
for col in categorical_columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x=train_df[col])
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.show()

## Correlation Matrix
plt.figure(figsize=(10,8))
sns.heatmap(train_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlations")
plt.show()

# Step 4: Define features and target variable
X = train_df.drop(columns=["id", "target"])
y = train_df["target"]

# Step 5: Split train-validation for model selection
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

## Feature Importance Plot
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh', figsize=(10,6))
plt.title("Top 10 Feature Importances")
plt.show()

# Step 7: Predict and evaluate on validation set
y_pred = model.predict(X_val)
f1 = f1_score(y_val, y_pred)
print(f"Validation F1-Score: {f1:.4f}")

## Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 8: Predict on test data
X_test = test_df.drop(columns=["id"])
test_predictions = model.predict(X_test)

# Step 9: Save predictions in required format
submission_df = pd.DataFrame({"id": test_df["id"], "target": test_predictions})
submission_df.to_csv("submissions.csv", index=False)

print("Submission file saved as submissions.csv")

```

---

## 3. Predct Restaurant Rating

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Step 2: Data Preprocessing
## Drop unnecessary columns
train_df.drop(columns=["id", "name", "location", "phone"], inplace=True)
test_df.drop(columns=["id", "name", "location", "phone"], inplace=True)

## Encode categorical variables
categorical_columns = ["table_bookings", "online_ordering", "restaurant_type", "restaurant_operation", "primary_cuisine"]
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

## Convert ratings to numerical values (extract numeric part)
def clean_rating(value):
    try:
        return float(str(value).split('/')[0])  # Extract only the numeric rating
    except ValueError:
        return np.nan

train_df["rating"] = train_df["rating"].apply(clean_rating)
test_df["rating"] = test_df["rating"].apply(clean_rating)

## Fill missing values with median
train_df.fillna(train_df.median(numeric_only=True), inplace=True)
test_df.fillna(test_df.median(numeric_only=True), inplace=True)

## Standardize numerical features
scaler = StandardScaler()
num_cols = ["popular_dishes", "cuisines_offered", "rating", "votes"]
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

# Step 3: Data Visualization
## Histograms of numerical features
train_df.hist(figsize=(12, 10))
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

## Categorical Feature Distribution
for col in categorical_columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x=train_df[col])
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.show()

## Correlation Matrix
plt.figure(figsize=(10,8))
sns.heatmap(train_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlations")
plt.show()

# Step 4: Define features and target variable
X = train_df.drop(columns=["dining_cost"])
y = train_df["dining_cost"]

# Step 5: Split train-validation for model selection
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

## Feature Importance Plot
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh', figsize=(10,6))
plt.title("Top 10 Feature Importances")
plt.show()

# Step 7: Predict and evaluate on validation set
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

## Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Budget", "Expensive"], yticklabels=["Budget", "Expensive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 8: Predict on test data
X_test = test_df.drop(columns=["dining_cost"], errors='ignore')
test_predictions = model.predict(X_test)

# Step 9: Save predictions in required format
submission_df = pd.DataFrame({"id": test_df.index, "dining_cost": test_predictions})
submission_df.to_csv("submissions.csv", index=False)

print("Submission file saved as submissions.csv")

```

---

## 4. Predct Candidate Compensation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Step 1: Load datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Step 2: Data Preprocessing
## Drop unnecessary columns
train_df.drop(columns=["id", "timestamp", "country", "job_title"], inplace=True)
test_df.drop(columns=["id", "timestamp", "country", "job_title"], inplace=True)

## Encode categorical variables
categorical_columns = ["employment_status", "is_manager", "education", "is_education_computer_related", "certifications"]
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

## Fill missing values with median
train_df.fillna(train_df.median(numeric_only=True), inplace=True)
test_df.fillna(test_df.median(numeric_only=True), inplace=True)

## Standardize numerical features
scaler = StandardScaler()
num_cols = ["job_years", "hours_per_week", "telecommute_days_per_week"]
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

# Step 3: Data Visualization
## Histograms of numerical features
train_df.hist(figsize=(12, 10))
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

## Categorical Feature Distribution
for col in categorical_columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x=train_df[col])
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.show()

## Correlation Matrix
plt.figure(figsize=(10,8))
sns.heatmap(train_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlations")
plt.show()

# Step 4: Define features and target variable
X = train_df.drop(columns=["salary"])
y = train_df["salary"]

# Step 5: Split train-validation for model selection
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train RandomForest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

## Feature Importance Plot
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh', figsize=(10,6))
plt.title("Top 10 Feature Importances")
plt.show()

# Step 7: Predict and evaluate on validation set
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
print(f"Validation MAE: {mae:.4f}")

# Step 8: Predict on test data
X_test = test_df.drop(columns=["salary"], errors='ignore')
test_predictions = model.predict(X_test)

# Step 9: Save predictions in required format
submission_df = pd.DataFrame({"id": test_df.index, "salary": test_predictions})
submission_df.to_csv("submissions.csv", index=False)

print("Submission file saved as submissions.csv")

```

---

## 5. Predct Auto Insurance

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Step 2: Data Preprocessing
## Drop unnecessary columns
train_df.drop(columns=["Customer", "State", "Effective Date"], inplace=True)
test_df.drop(columns=["Customer", "State", "Effective Date"], inplace=True)

## Encode categorical variables
categorical_columns = ["Coverage", "Education", "EmploymentStatus", "Gender", "Location Code", "Policy", "Sales Channel", "Vehicle Class", "Vehicle Size"]
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

## Fill missing values with median
train_df.fillna(train_df.median(numeric_only=True), inplace=True)
test_df.fillna(test_df.median(numeric_only=True), inplace=True)

## Standardize numerical features
scaler = StandardScaler()
num_cols = ["Customer Lifetime Value", "Income", "Monthly Premium Auto", "Months Since Policy Inception", "Number of Open Complaints", "Number of Policies", "Total Claim Amount"]
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

# Step 3: Data Visualization
## Histograms of numerical features
train_df.hist(figsize=(12, 10))
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

## Categorical Feature Distribution
for col in categorical_columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x=train_df[col])
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.show()

## Correlation Matrix
plt.figure(figsize=(10,8))
sns.heatmap(train_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlations")
plt.show()

# Step 4: Define features and target variable
X = train_df.drop(columns=["Response"])
y = train_df["Response"]

# Step 5: Split train-validation for model selection
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

## Feature Importance Plot
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh', figsize=(10,6))
plt.title("Top 10 Feature Importances")
plt.show()

# Step 7: Predict and evaluate on validation set
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

## Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 8: Predict on test data
X_test = test_df.drop(columns=["Response"], errors='ignore')
test_predictions = model.predict(X_test)

# Step 9: Save predictions in required format
submission_df = pd.DataFrame({"id": test_df.index, "Response": test_predictions})
submission_df.to_csv("submissions.csv", index=False)

print("Submission file saved as submissions.csv")

```

---

```python
'''
Calculating missing values for each column.'''
data.isna().mean()*100

'''
Imputing the missing categorical values with the most
frequent value.
Imputing the missing numerical values with the column
mean.
'''
for column, data_type in data.dtypes.items():
    if data_type == 'object':
        data[column] = data[column].fillna(data[column].mode().iloc[0])
    else:
        data[column] = data[column].fillna(data[column].mean())

'''Dropping features which have least variance is a good idea.'''
variance = data.var().reset_index()
variance

del data['Number of Open Complaints']


coverage=data.groupby('Coverage').agg({
    'Total Claim Amount':'median',
    'Monthly Premium Auto':'median'
    
})
coverage=coverage.reset_index().set_index('Coverage')
coverage.plot(kind='bar')
plt.show()

coverage=data.groupby('Coverage').agg({
    'Total Claim Amount':'median',
    'Monthly Premium Auto':'median'
    
})
coverage=coverage.reset_index().set_index('Coverage')
coverage['ratio']=coverage['Total Claim Amount']/coverage['Monthly Premium Auto']
coverage[['ratio']].plot(kind='bar')
plt.show()

coverage
"""
Customer Lifetime Value is higher for Premium policy as compared to extended or basic policy.The premium also shows a similar trend.But if we compare Claim to Premium ratio,they are very close and hence it seems that agency will be more profitable if it sells premium policies more as compared to basic.
"""

data.groupby('Coverage')['Customer Lifetime Value'].median().plot(kind='bar')
plt.show()


data.groupby(['Sales Channel'])['Monthly Premium Auto'].mean().plot(kind='bar',figsize=(10,7))
plt.show()

data.groupby(['Sales Channel'])['Response'].mean().plot(kind='bar')
plt.show()
"""
Policies which are sold through agent have higher success rate as compared to other channels.
"""


'''
Categorical columns wrt default status.
'''
sns.boxplot(data = data, x = 'Response', y = 'Monthly Premium Auto')
plt.show()

sns.boxplot(data = data, x = 'Response', y = 'Income')
plt.xticks(rotation=90)
plt.show()

sns.boxplot(data = data, x = 'Number of Policies', y = 'Income')
plt.xticks(rotation=90)
plt.show()

sns.pairplot(data)
plt.show()

data['Income'].plot(kind='hist')
plt.show()

data['Income'].describe([0.95,0.97,0.98])

data['Total Claim Amount'].plot(kind='hist')
plt.show()

data['Total Claim Amount'].describe([0.95,0.97,0.98,0.99])

data.loc[data['Total Claim Amount']>1500,'Total Claim Amount']=1500
data['Total Claim Amount'].plot(kind='hist')
plt.show()

"""
1. Claim Amount contains some outliers. Top 2 percentile of the distribution is significantly higher.Hence we imputed them.
2. Top 2 percentile income have significantly higher values than the rest.
3. Customers with 2 policies have highest lifetime values.
"""

correlation_df=data.drop(['Customer'], axis = 1).corr()
correlation_df


sns.heatmap(correlation_df, vmax = 1, vmin = -1)
plt.show()


data['Policy_type']=data['Policy'].str.split().str[0]
data['Policy_level']=data['Policy'].str[-1].astype(int)
del data['Policy']


coverage_map={'Basic':0,'Extended':1,'Premium':2}
data['Coverage']=data['Coverage'].map(coverage_map)

education_map={'High School or Below':0,'College':1,'Bachelor':2,'Master':3,'Doctor':4}
data['Education']=data['Education'].map(education_map)

'''
Creating one hot encoded features from categorical columns
'''
data=pd.get_dummies(data,columns=['Policy_type','State','EmploymentStatus','Gender','Location Code',
                                  'Sales Channel','Vehicle Class'])

'''
Creating features from date
'''
data['Effective Date']=pd.to_datetime(data['Effective Date'])
data['Effective Date_week'] = data['Effective Date'].dt.isocalendar().week
for n in ('Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear',
        'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
    data['Effective Date_'+n] = getattr(data['Effective Date'].dt,n.lower())
    
del data['Effective Date']

'''
Using Transformations on numerical features
'''
clv=MinMaxScaler()
clv.fit(data['Customer Lifetime Value'].to_numpy().reshape(-1,1))
data['Customer Lifetime Value']=clv.transform(data['Customer Lifetime Value'].to_numpy().reshape(-1,1))
data['Income']=np.log(1+data['Income'])

data['value_per_policy_per_month']=data['Customer Lifetime Value']/(data['Number of Policies']
                                                                    *data['Months Since Policy Inception'])
data['lifetime_years']=data['Customer Lifetime Value']/data['Monthly Premium Auto']

data.head()

'''
Preprocessing on the test testset.
'''
test = pd.read_csv("test.csv")
test['Policy_type']=test['Policy'].str.split().str[0]
test['Policy_level']=test['Policy'].str[-1].astype(int)
del test['Policy'],test['Number of Open Complaints']

test['Coverage']=test['Coverage'].map(coverage_map)

test['Education']=test['Education'].map(education_map)

test=pd.get_dummies(test,columns=['Policy_type','State','EmploymentStatus','Gender','Location Code',
                                  'Sales Channel','Vehicle Class'])

test['Effective Date']=pd.to_datetime(test['Effective Date'])
test['Effective Date_week'] = test['Effective Date'].dt.isocalendar().week
for n in ('Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear',
        'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
    test['Effective Date_'+n] = getattr(test['Effective Date'].dt,n.lower())
    
del test['Effective Date']

test['Customer Lifetime Value']=clv.transform(test['Customer Lifetime Value'].to_numpy().reshape(-1,1))

test['Income']=np.log(1+test['Income'])
test['value_per_policy_per_month']=test['Customer Lifetime Value']/(test['Number of Policies']
                                                                    *test['Months Since Policy Inception'])
test['lifetime_years']=test['Customer Lifetime Value']/test['Monthly Premium Auto']


test.head()

X_train,X_valid,Y_train,Y_valid=train_test_split(
    data.drop(['Customer','Response'], axis = 1),
    data[['Response']],
    train_size=0.8)

model = XGBClassifier()
X_train = X_train.astype({'Effective Date_week': 'uint32'})
X_train.replace([np.inf, -np.inf], 0, inplace=True)
model.fit(X_train, Y_train)

X_valid = X_valid.astype({'Effective Date_week': 'uint32'})
Y_pred=model.predict(X_valid)

accuracy_score(Y_valid,Y_pred)

params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

xgb = XGBClassifier(learning_rate=0.02, n_estimators=1000, objective='binary:logistic',
                    silent=True, nthread=1)

random_search = RandomizedSearchCV(xgb, 
                                   param_distributions=params,
                                   n_iter=5, 
                                   scoring='accuracy', 
                                   n_jobs=4,
                                cv=5)
random_search.fit(X_train, Y_train)

best_model=random_search.best_estimator_
Y_pred=best_model.predict(X_valid)
accuracy_score(Y_valid,Y_pred)


feature_imp=pd.DataFrame()
for feature,imp in zip(X_train.columns,best_model.feature_importances_):
    temp=pd.DataFrame([feature,imp]).T
    feature_imp=feature_imp.append(temp)
feature_imp.columns=['feature','relative_importance']
feature_imp.sort_values(by='relative_importance',inplace=True)
feature_imp.set_index('feature',inplace=True)
feature_imp.iloc[-20:,:].plot(kind='barh',figsize=(10,8))
plt.show()


test = test.astype({'Effective Date_week': 'uint32'})
submission=best_model.predict(test.drop('Customer',axis=1))


submission_df=pd.DataFrame(
{
    'Customer':test.Customer,
    'Response':submission
})


submission_df.to_csv('submission.csv',index=False)

```
