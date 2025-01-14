# Pushing a CSV File into PostgreSQL using Jupyter Notebook

This guide provides step-by-step instructions to load a CSV file into a PostgreSQL table using a Jupyter Notebook.

---

## Prerequisites

Ensure the following Python libraries are installed:

```bash
pip install pandas sqlalchemy psycopg2
```

---

## Steps

### 1. Import Required Libraries

Open a Jupyter Notebook and import the necessary libraries:

```python
import pandas as pd
from sqlalchemy import create_engine
```

---

### 2. Load the CSV File

Replace `"your_file.csv"` with the path to your CSV file.

```python
# Load the CSV file
csv_file_path = "your_file.csv"  # Replace with your file path
df = pd.read_csv(csv_file_path)

# Preview the first few rows
print(df.head())
```

---

### 3. Create the Database in pgAdmin

1. Navigate to [http://localhost:5050/](http://localhost:5050/) and log in using your credentials.
   - **Email**: `admin@example.com` (or your configured email).
   - **Password**: `admin` (or your configured password).

2. Add a new database:
   - Right-click on **Servers > PostgreSQL > Databases**.
   - Select **Create > Database...**.
   - Name the database (e.g., `mydb`).
   - Save it.

---

### 4. Set Up PostgreSQL Connection

Replace the placeholders with your PostgreSQL connection details:

- `postgres_user`: Your PostgreSQL username (e.g., `postgres`).
- `postgres_password`: Your PostgreSQL password (e.g., `password`).
- `db_name`: The name of your database (e.g., `mydb`).
- `host`: The hostname of your database (e.g., `localhost` or `db` if using Docker).

```python
# Define connection details
postgres_user = "postgres"
postgres_password = "password"
db_name = "mydb"
host = "localhost"  # or "db" if running in Docker

# Create an SQLAlchemy engine
engine = create_engine(f"postgresql+psycopg2://{postgres_user}:{postgres_password}@{host}:5432/{db_name}")
```

---

### 5. Push Data to PostgreSQL

Provide the name of the target table in PostgreSQL (e.g., `my_table`). If the table doesn’t exist, it will be created.

```python
# Define the table name
table_name = "my_table"

# Push the DataFrame to PostgreSQL
df.to_sql(table_name, engine, if_exists="replace", index=False)

print(f"Data successfully pushed to the '{table_name}' table in PostgreSQL!")
```

---

### 6. Verify Data in PostgreSQL

Use pgAdmin or a SQL query tool to verify the data was successfully inserted:

```sql
SELECT * FROM my_table LIMIT 10;
```

---

# Pulling a DataFrame from PostgreSQL

This guide provides step-by-step instructions to pull a DataFrame from a PostgreSQL table using a Jupyter Notebook.

---

### 1. Import Required Libraries

Ensure you have the necessary libraries installed and imported:

```python
import pandas as pd
from sqlalchemy import create_engine
```

---

### 2. Set Up PostgreSQL Connection

Replace the placeholders with your PostgreSQL connection details:

```python
# Define connection details
postgres_user = "postgres"
postgres_password = "password"
db_name = "mydb"
host = "localhost"  # or "db" if running in Docker

# Create an SQLAlchemy engine
engine = create_engine(f"postgresql+psycopg2://{postgres_user}:{postgres_password}@{host}:5432/{db_name}")
```

---

### 3. Pull Data from PostgreSQL

Specify the table name or write a SQL query to retrieve the data:

```python
# Define the table name or SQL query
table_name = "my_table"  # Replace with your table name
query = f"SELECT * FROM {table_name};"

# Execute the query and load data into a DataFrame
df = pd.read_sql(query, engine)

# Preview the retrieved data
print("Preview of the retrieved data:")
print(df.head())
```

---

### 4. Work with the DataFrame

Once the data is loaded into the DataFrame, you can perform any analysis or processing tasks in Python as needed.

---

## Full Notebook Code Example (Pull Data)

Here’s the complete code block to pull data from PostgreSQL:

```python
import pandas as pd
from sqlalchemy import create_engine

# Define PostgreSQL connection details
postgres_user = "postgres"
postgres_password = "password"
db_name = "mydb"
host = "localhost"  # or "db" if running in Docker

# Create an SQLAlchemy engine
engine = create_engine(f"postgresql+psycopg2://{postgres_user}:{postgres_password}@{host}:5432/{db_name}")

# Define the table name or SQL query
table_name = "my_table"  # Replace with your table name
query = f"SELECT * FROM {table_name};"

# Execute the query and load data into a DataFrame
df = pd.read_sql(query, engine)

# Preview the retrieved data
print("Preview of the retrieved data:")
print(df.head())
```

---

## Notes

- Ensure the table name in the query matches the name in your PostgreSQL database.
- For large datasets, you can use the `chunksize` parameter in `pd.read_sql()` to load data in chunks.
- Use this DataFrame for any further analysis or visualization tasks.

---
