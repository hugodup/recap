# Python Dataset Extraction and Analysis Cheat Sheet

This cheat sheet provides step-by-step instructions for extracting and analyzing datasets from various file types using Python.

---

## 1. **Setup and Installation**

```bash
# Install necessary libraries
pip install pandas numpy matplotlib seaborn openpyxl xlrd pyarrow fastparquet plotly
```

---

## 2. **Import Libraries**

```python
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
```

---

## 3. **Read Data**

### a. CSV Files
```python
data = pd.read_csv('file.csv')
```

### b. Excel Files
```python
data = pd.read_excel('file.xlsx', engine='openpyxl')
```

### c. JSON Files
```python
data = pd.read_json('file.json')
```

### d. Parquet Files
```python
data = pd.read_parquet('file.parquet', engine='pyarrow')
```

### e. Text Files (e.g., TSV)
```python
data = pd.read_csv('file.txt', sep='\t')
```

---

## 4. **Quick Look at the Data**

### a. View Top Rows
```python
data.head()
```

### b. View Bottom Rows
```python
data.tail()
```

### c. Get Basic Information
```python
data.info()
```

### d. Summary Statistics
```python
data.describe()
```

### e. Check for Missing Values
```python
data.isnull().sum()
```

---

## 5. **Data Cleaning**

### a. Drop Missing Values
```python
data = data.dropna()
```

### b. Fill Missing Values
```python
data.fillna(value, inplace=True)
```

### c. Rename Columns
```python
data.rename(columns={'old_name': 'new_name'}, inplace=True)
```

### d. Drop Duplicates
```python
data = data.drop_duplicates()
```

---

## 6. **Basic Analysis**

### a. Column Selection
```python
selected_column = data['column_name']
```

### b. Filtering Rows
```python
filtered_data = data[data['column_name'] > value]
```

### c. Grouping Data
```python
grouped = data.groupby('column_name').mean()
```

### d. Sorting Data
```python
sorted_data = data.sort_values(by='column_name', ascending=True)
```

---

## 7. **Visualization**

### a. Histogram
```python
fig = px.histogram(data, x='column_name')
fig.show()
```

### b. Scatter Plot
```python
fig = px.scatter(data, x='x_column', y='y_column')
fig.show()
```

### c. Correlation Heatmap
```python
corr = data.corr()
fig = px.imshow(corr, text_auto=True, color_continuous_scale='coolwarm')
fig.show()
```

### d. Box Plot
```python
fig = px.box(data, x='category_column', y='value_column')
fig.show()
```

---

## 8. **Export Processed Data**

### a. Save as CSV
```python
data.to_csv('output.csv', index=False)
```

### b. Save as Excel
```python
data.to_excel('output.xlsx', index=False)
```

### c. Save as JSON
```python
data.to_json('output.json', orient='records')
```

### d. Save as Parquet
```python
data.to_parquet('output.parquet', engine='pyarrow', index=False)
```

---

## 9. **Additional Tips**

- Use `.sample(n=5)` to randomly preview 5 rows of data.
- Use `data.columns` to list all column names.
- Chain operations for streamlined workflows, e.g.,

```python
cleaned_data = (data.dropna()
                   .rename(columns={'old_name': 'new_name'})
                   .sort_values(by='column_name'))
```

---

## 10. **Debugging Common Issues**

- **File not found**: Check the file path and ensure it is correct.
- **Missing library**: Install the required library using `pip install`.
- **Data type issues**: Use `data.dtypes` to inspect column types and convert if necessary:

```python
data['column_name'] = data['column_name'].astype('int')
```

---

## 11. **Example**

```python
def df_cleaning_nrg_bal_s(df_, unit_, nrg_, siec_, energy_simplified=energy_siec_simplified,energy_balances = energy_nrg_bal_simplified):
    
    """
    Function to clean the dataset: nrg_bal_s
        df_:     dataframe
        date_:   name of the date column
        unit_: name of the metric1 column
    """
    # Convert variable to list
    # Check if the variable is a list
    if type(nrg_) != list:
        nrg_ = [nrg_]
    if type(siec_) != list:
        siec_ = [siec_]
    if type(unit_) != list:
        unit_ = [unit_]

    # Filter the DataFrame based on column values
    df_ = df_.copy()
    df_ = df_[df_['nrg_bal'].isin(nrg_)]
    df_ = df_[df_['siec'].isin(siec_)]

    # convert GHW to TWH
    if unit_ == ['TWH']:
        df_ = df_[df_['unit'].isin(['GWH'])]
        
    else: 
        df_ = df_[df_['unit'].isin(unit_)]


    # Converting the time to date + only year
    df_['Year'] = pd.to_datetime(df_['time'], format='%Y')
    df_['Year'] = df_['Year'].dt.year  # Extract only the year from the 'date' column
    df_['Year'] = df_['Year'].astype(str)
    df_['Year'] = df_['Year'].str.extract('([-+]?\d*\.?\d+)').astype('int64') # Making sure to clean all fields from unexpected char
    

    # Getting the last 10 years
    max_date = df_['Year'].max() # Find the maximum date
    start_date = max_date - 10 # Calculate the starting point for the last 10 years
    df_ = df_[df_['Year'] > start_date] # Filter the DataFrame for the last 10 years
    
    # cleaning the data from stings inside
    df_['values'] = df_['values'].astype(str)
    df_['values'] = df_['values'].str.extract('([-+]?\d*\.?\d+)').astype(float)

    # Converting to GWH to TWJ
    if unit_ == ['TWH']:
        df_['values'] = df_['values']/1000
        unit_ = ['TWH']
        
    else: 
        pass

    # API error code:
    countries_to_correct = {
        'EL': 'GR',
        'RS_ME': 'Serbia-Montenegro',
        'XK': 'Kosovo',
        'EU27_2020': 'EU'
    }
    df_['geo'] = df_['geo'].astype(str).replace(countries_to_correct)

    # Add country name to each country code
    df_['country'] = df_['geo'].apply(get_country_name)

    # Filtering the EU countries
    df_ = df_[df_['country'].isin(eu_countries)]
    df_ = df_[df_['country'] != '']
    # print(df_)

    # Mapping Energy SIEC & Balances  from large dictonary of definitions
    df_['siec'] = df_['siec'].astype(str).replace(energy_simplified)
    df_['nrg_bal'] = df_['nrg_bal'].astype(str).replace(energy_balances)
    
    # columns to drop
    cols_to_drop = ['freq', 'unit', 'geo', 'time']
    df_ = df_.drop(cols_to_drop, axis=1)

    # filtering
    excluded_values = ['EU28', 'EU27_2007', 'EU25', 'EU15', 'EFTA', 'UNK', 'WORLD']
    df_ = df_[~df_['country'].isin(excluded_values)]  

    # Generating the shares for each year
    # Create a new column 'TJ (%)' per Year with the share per year
    name = unit_[0] + " (%)"
    total_by_year = df_.groupby(['Year', 'country'])['values'].sum()
    df_[name] = df_['values'] / df_.apply(lambda row: total_by_year.loc[(row['Year'], row['country'])], axis=1) * 100
    

    # Reordering columns
    df_ = df_[['Year', 'country', 'nrg_bal', 'siec', 'values', name]]
    # df_ = df_[['Year', 'country', 'nrg_bal', 'siec', 'values']]

    # Renaming columns
    df_ = df_.rename(columns={'values': unit_[0]})
   
    # Reset index
    # Sorting
    df_ = df_.sort_values(['country', 'Year', 'nrg_bal', 'siec'], ascending=[True, True, True, True]).reset_index(drop=True)

    # Return df_
    return df_
```
