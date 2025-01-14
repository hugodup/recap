# Cheat Sheet: Building a Robust Data Pipeline with Docker, PostgreSQL, and Python

## **Overview**
This guide outlines the steps to create a data pipeline for retrieving, processing, and storing data using Python, Docker, PostgreSQL, and supporting tools.

---

## **Key Components**
- **Python**: For scripting the data processing tasks.
- **Docker**: To containerize the application for portability.
- **PostgreSQL**: As the relational database management system.
- **PgAdmin & pgcli**: Tools for managing and interacting with PostgreSQL.

---

## **1. Directory Structure**

```
project-folder/
|-- Dockerfile
|-- requirements.txt
|-- main.py
|-- data/
     |-- extracted_data.csv
```

---

## **2. Creating Docker Network**
Facilitates communication between the database and other services:
```bash
docker network create chicago_crime_network
```

---

## **3. PostgreSQL Docker Container**
Run PostgreSQL with environment variables and data volume:
```bash
docker run -it \
    -e POSTGRES_USER=root \
    -e POSTGRES_PASSWORD=root \
    -e POSTGRES_DB=chicago_crime_data \
    -v "$(pwd)/chicago_crime_data:/var/lib/postgresql/data" \
    -p 5432:5432 \
    --name chicago_crime_data_database \
    --network=chicago_crime_network \
    postgres:13
```

---

## **4. Dockerfile for Pipeline**
Define the Docker image:

```dockerfile
# Use official Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN apt-get update && apt-get install -y wget \
    && pip install -r requirements.txt

# Entry point
ENTRYPOINT ["python", "main.py"]
```

---

## **5. Python Script (main.py)**

### **Functions**
- **`read_data_from_csv`**: Reads and processes CSV files.
- **`manipulate_data`**: Performs transformations.
- **`push_to_db`**: Writes processed data to PostgreSQL.

```python
def read_data_from_csv(file_path):
    # Logic to read data
    pass

def manipulate_data(data):
    # Transform data
    pass

def push_to_db(data, engine, table_name):
    # Push data in chunks
    pass

if __name__ == "__main__":
    # Execute the pipeline
    pass
```

---

## **6. Running the Pipeline**
Execute the Python script with database credentials:
```bash
python pipeline.py \
    --user=root \
    --password=root \
    --host=localhost \
    --port=5432 \
    --db=chicago_crime_data \
    --table_name=chicago_crime_data
```

---

## **7. Build and Run Docker Image**

### Build Docker Image
```bash
docker build -t crime_data .
```

### Run Docker Container
```bash
docker run --network=chicago_crime_network crime_data
```

---

## **8. Tools for Interactions**

### pgcli (PostgreSQL CLI):
```bash
pgcli -h localhost -p 5432 -U root -d chicago_crime_data
```

### PgAdmin:
Access PgAdmin via browser to manage the database.

---

## **Summary**
This pipeline automates the daily processing and storage of data using a robust architecture. Use this cheat sheet to quickly set up and run your own pipeline with Docker and PostgreSQL.