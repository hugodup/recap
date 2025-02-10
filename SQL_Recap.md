# SQL Advanced Cheat Sheet

## Querying Tables with SELECT
- Fetch all columns from the `customers` table:
```sql
SELECT *
FROM customer;
```
- Fetch `name` and `age` columns for all customers, sorted by `age` in ascending order:
```sql
SELECT name, age
FROM customers
ORDER BY age ASC;
```

- Sort customers by age in descending order (high to low):
```sql
SELECT *
FROM customers
ORDER BY age DESC;
```

### Aliases
- Assign alias to column:
```sql
SELECT cost * 0.04 AS sales_tax
FROM orders;
```
- Assign alias to table:
```sql
SELECT cus.name, cus.age
FROM customers AS cus;
```

## Filtering Output with WHERE
- Fetch customers over the age of 35:
```sql
SELECT *
FROM customers
WHERE age > 35;
```
- Fetch customers from `USA` or `Canada` with a subscription:
```sql
SELECT *
FROM customers
WHERE (country = 'USA' OR country = 'Canada') AND has_subscription = TRUE;
```
- Fetch customers where the city starts with "New":
```sql
SELECT *
FROM customers
WHERE city LIKE 'New%';
```
- Fetch customers living in North America:
```sql
SELECT name
FROM customers
WHERE country IN ('USA', 'Canada', 'Mexico');
```

## Joining Tables
- `INNER JOIN`:
```sql
SELECT orders.order_id, orders.cus_id, customers.id, customers.name
FROM orders
INNER JOIN customers
ON orders.cus_id = customers.id;
```
- `FULL OUTER JOIN`:
```sql
SELECT orders.order_id, customers.name
FROM orders
FULL OUTER JOIN customers
ON orders.cus_id = customers.id;
```
- `LEFT JOIN`:
```sql
SELECT customers.name, orders.date, orders.cost
FROM customers
LEFT JOIN orders
ON customers.id = orders.cus_id;
```
- `RIGHT JOIN`:
```sql
SELECT orders.order_id, customers.name
FROM orders
RIGHT JOIN customers
ON orders.cus_id = customers.id;
```

## Aggregation and Grouping
- Basic Aggregations:
```sql
SELECT cus_id,
    SUM(cost) AS sum_cost,
    COUNT(order_id) AS count_id,
    MAX(cost) AS max_cost,
    ROUND(AVG(cost), 2) AS avg_cost
FROM orders
GROUP BY cus_id
ORDER BY cus_id;
```

## Subqueries
- Subquery returning a single value:
```sql
SELECT order_id
FROM orders
WHERE cost > (
    SELECT AVG(cost)
    FROM orders
);
```
- Subquery returning multiple values:
```sql
SELECT order_id
FROM orders
WHERE cus_id = ANY (
    SELECT id AS cus_id
    FROM customers
    WHERE country = 'USA');
```

## Window Functions
- Ranking rows based on cost:
```sql
SELECT order_id, cost,
RANK() OVER (
    ORDER BY cost DESC
) AS order_rank
FROM orders;
```

- Partitioning rows by `cus_id` and ordering by cost:
```sql
SELECT order_id, cus_id, cost,
SUM(cost) OVER (PARTITION BY cus_id ORDER BY cost ASC) AS sum_cost
FROM orders;
```

## Case Statements
- Categorize orders by cost:
```sql
SELECT order_id, cus_id, cost,
CASE
    WHEN cost > 175 THEN 'luxury'
    WHEN cost > 100 THEN 'mid-tier'
    ELSE 'budget'
END AS product_type
FROM orders;
```

## Set Operations
- Combine rows from multiple queries with `UNION` (removes duplicates):
```sql
SELECT name
FROM actors
WHERE country = 'Canada'
UNION
SELECT name
FROM singers
WHERE country = 'Canada';
```

## Common Table Expressions (CTEs)
- Simplify queries with temporary result sets:
```sql
WITH sum_sales AS (
    SELECT cus_id, SUM(cost) AS tot_sales
    FROM orders
    GROUP BY cus_id
)
SELECT cus_id, tot_sales
FROM sum_sales
WHERE tot_sales > 350;
```

## Useful Functions
- String functions: `LENGTH()`, `TRIM()`, `CONCAT()`
- Date functions: `NOW()`
- Rounding: `CEILING()`, `FLOOR()`
- Handling NULLs: `COALESCE()`
- Type conversion: `CAST()`

## Performance Tuning and Query Optimization
- Using Execution Plan:
```sql
EXPLAIN SELECT * FROM Employees WHERE department = 'IT';
```
- Optimizing Queries:
  - **Use Indexing**
  ```sql
  CREATE INDEX idx_department ON Employees(department);
  ```
  - **Avoid SELECT ***
  ```sql
  SELECT name, salary FROM Employees;
  ```
  - **Use Joins Instead of Subqueries**

## Prepared Statements
```sql
PREPARE stmt FROM 'SELECT * FROM Employees WHERE department = ?';
SET @dept = 'IT';
EXECUTE stmt USING @dept;
DEALLOCATE PREPARE stmt;
```

## Data Integrity and Constraints
```sql
CREATE TABLE Employees (
    id INT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    salary DECIMAL(10,2) CHECK (salary > 0),
    department_id INT,
    CONSTRAINT fk_department FOREIGN KEY (department_id) REFERENCES Departments(id)
);
```
