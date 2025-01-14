# Building a Data Pipeline with Docker, PostgreSQL, and Jupyter Notebook

This tutorial guides you through creating a data pipeline using Docker for PostgreSQL and pgAdmin, and running Jupyter Notebook locally.

---

## Updated Project Structure

```
postgres_pipeline/
├── docker-compose.yml
├── data/
└── notebooks/
```

---

## Step 1: Setting Up Docker for PostgreSQL and pgAdmin

### 1. Create `docker-compose.yml`

Create a `docker-compose.yml` file in the `postgres_pipeline/` directory with the following content:

```yaml
version: '3.9'

services:
  postgres:
    image: postgres:latest
    container_name: postgres
    restart: always
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: parking_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql

  pgadmin:
    image: dpage/pgadmin4
    container_name: pgadmin
    restart: always
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@example.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "8080:80"

  jupyter:
    build: .
    container_name: jupyter
    restart: always
    volumes:
      - ./jupyter_notebook:/app/jupyter_notebook
    ports:
      - "8888:8888"

volumes:
  postgres_data:

```

### 2. Start the Services

Run the following command to start PostgreSQL and pgAdmin:

```bash
docker-compose up --build
```

- **PostgreSQL** will run on `localhost:5432`.
- **pgAdmin** will be accessible at [http://localhost:5050](http://localhost:5050).

### 3. Configure pgAdmin

1. Navigate to [http://localhost:5050](http://localhost:5050) and log in with:
   - **Email**: `admin@example.com`
   - **Password**: `admin`

2. Add a new server:
   - Click on **Add New Server**.
   - **General Tab**: Enter a name (e.g., `Postgres`).
   - **Connection Tab**: Fill in the following details:
     - **Host name/address**: `postgres`
     - **Port**: `5432`
     - **Username**: `postgres`
     - **Password**: `password`
   - Click **Save**.

---

## Step 2: Install Jupyter Notebook Locally

### 1. Install Jupyter and Required Libraries

On your local machine, install the following:

```bash
pip install jupyter pandas sqlalchemy psycopg2 kagglehub
```

### 2. Start Jupyter Notebook

1. Navigate to the `notebooks/` directory:

   ```bash
   cd postgres_pipeline/notebooks
   ```

2. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Access Jupyter Notebook at [http://localhost:8888](http://localhost:8888).

---

## Step 3: Download and Push the Dataset to PostgreSQL

```python
!pip install pandas sqlalchemy kagglehub psycopg2
```

```python
import pandas as pd
from sqlalchemy import create_engine
import kagglehub
```

```python
# Download the dataset using KaggleHub
csv_file_path = kagglehub.dataset_download("mfaisalqureshi/parking")

print("Path to dataset files:", csv_file_path)

# Load the dataset into a Pandas DataFrame
df = pd.read_csv(csv_file_path + "/" + "Parking Data.csv")

# Preview the dataset
print(df.head())
```
---

## Step 4: Connect Jupyter to PostgreSQL

```python
from sqlalchemy import create_engine

# Define connection details
postgres_user = "postgres"
postgres_password = "password"
db_name = "pipeline_db"
host = "postgres"  # Use 'localhost' if running outside Docker

# Create the SQLAlchemy engine
engine = create_engine(f"postgresql+psycopg2://{postgres_user}:{postgres_password}@{host}:5432/{db_name}")
```

```python
# Define the table name
table_name = "parking_data"

# Push the DataFrame to PostgreSQL
df.to_sql(table_name, engine, if_exists="replace", index=False, method="multi")

print(f"Data successfully pushed to the '{table_name}' table!")
```

---

## Step 5: Verify the Data in pgAdmin

1. Open pgAdmin at [http://localhost:5050](http://localhost:5050).
2. Navigate to the `pipeline_db` database and open the **Query Tool**.
3. Run the query:

   ```sql
   SELECT * FROM parking_data LIMIT 10;
   ```

You should see the dataset successfully imported into PostgreSQL.

---

## Step 6: Query Parking Data from Jupyter Notebook

Once the data is loaded into PostgreSQL, you can query it directly from your Jupyter Notebook using Python.

```python
# Query the parking data from PostgreSQL
query = "SELECT * FROM parking_data LIMIT 10;"
df_parking = pd.read_sql(query, engine)

# Preview the queried data
print("Queried data:")
print(df_parking.head())
```

This allows you to integrate your parking data with further analysis or processing tasks directly in Python.

---

## Next Steps

- Automate the pipeline to handle regular updates.
- Add validation or preprocessing steps to clean the dataset before pushing it to PostgreSQL.
- Use visualization tools like **Matplotlib** or **Seaborn** in Jupyter Notebook for further analysis.

---

You're all set to build and use your data pipeline! 🚀
