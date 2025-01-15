# Python Dates and Times Cheat Sheet

## Key Definitions

- **Date**: Handles dates without time.
- **POSIXct**: Handles date & time in calendar time.
- **POSIXlt**: Handles date & time in local time.
- **Hms**: Parses periods with hour, minute, and second.
- **Timestamp**: Represents a single pandas date & time.
- **Interval**: Defines an open or closed range between dates and times.
- **Timedelta**: Computes time differences between different datetimes.

## Advantages of ISO 8601 Format

- Specifies datetimes as `YYYY-MM-DD HH:MM:SS TZ`.
- Avoids ambiguities between different date formats.
- Language-independent numeric month values (e.g., 08 instead of AUG).
- Optimized for comparisons and sorting in Python.

---

## Common Libraries

```python
import datetime as dt
import time as tm
import pytz
import pandas as pd
```

---

## Getting the Current Date and Time

```python
# Current date
current_date = dt.date.today()

# Current date and time
current_datetime = dt.datetime.now()
```

---

## Reading Date, Datetime, and Time Columns from CSV

```python
# Specify datetime columns
pd.read_csv("filename.csv", parse_dates=["col1", "col2"])

# Specify datetime from components
pd.read_csv("filename.csv", parse_dates={"datetime": ["year", "month", "day"]})
```

---

## Parsing Dates and Times

```python
# Parse ISO format
pd.to_datetime(iso)

# Parse US format
pd.to_datetime(us, dayfirst=False)

# Parse non-US format
pd.to_datetime(non_us, dayfirst=True)

# Infer datetime format
pd.to_datetime(iso, infer_datetime_format=True)

# Specify datetime format
pd.to_datetime(us, format="%m/%d/%Y %H:%M:%S")
```

---

## Extracting Components

```python
dttm = pd.to_datetime(iso)

# Year
dttm.dt.year

# Day of the year
dttm.dt.day_of_year

# Month name
dttm.dt.month_name()

# Day name
dttm.dt.day_name()

# Convert to datetime.datetime
dttm.dt.to_pydatetime()
```

---

## Time Zones

```python
# Get current time zone
tm.localtime().tm_zone

# Get a list of all time zones
pytz.all_timezones

# Localize datetime with a timezone
dttm.dt.tz_localize('CET')

# Convert datetime to another timezone
dttm.dt.tz_convert('US/Central')
```

---

## Rounding Dates

```python
# Round to nearest time unit
dttm.dt.round('1min')

# Floor to nearest time unit
dttm.dt.floor('1min')

# Ceil to nearest time unit
dttm.dt.ceil('1min')
```

---

## Arithmetic with Dates and Times

```python
# Create two datetimes
now = dt.datetime.now()
then = pd.Timestamp('2021-09-15 10:03:30')

# Time difference
time_elapsed = now - then

# Time elapsed in seconds
time_elapsed_seconds = time_elapsed.total_seconds()

# Add a day to a datetime
day_added = now + dt.timedelta(days=1)
```

---

## Time Intervals

```python
# Create interval datetimes
start_1 = pd.Timestamp('2021-10-21 03:02:10')
finish_1 = pd.Timestamp('2022-09-15 10:03:30')

# Specify interval
interval = pd.Interval(start_1, finish_1, closed='right')

# Interval length
interval_length = interval.length

# Check if intervals overlap
interval_overlap = interval.overlaps(pd.Interval(start_2, finish_2, closed='right'))

# Generate a range of intervals
intervals = pd.interval_range(start=pd.Timestamp('2017-01-01'), periods=3, freq='MS')
```

---

## Time Deltas

```python
# Define a timedelta
seven_days = pd.Timedelta(7, unit='d')

# Convert timedelta to seconds
seven_days_seconds = seven_days.total_seconds()
```

---

## Guessing Date Formats

```python
from pandas.tseries.api import guess_datetime_format

# Guess datetime format
date_format = guess_datetime_format('09/13/2023')  # Output: '%m/%d/%Y'
```

---

## Generating Date Ranges

```python
# Generate a range of dates
date_range = pd.date_range(start='2022-01-01', end='2022-01-10')

# Generate a range with a specific frequency
date_range_freq = pd.date_range(start='2022-01-01', periods=5, freq='D')

# Generate business days only
business_days = pd.bdate_range(start='2022-01-01', periods=5)

# Generate month starts
month_starts = pd.date_range(start='2022-01-01', periods=3, freq='MS')

# Generate quarter starts
quarter_starts = pd.date_range(start='2022-01-01', periods=3, freq='QS')
```

---

## Example: Calculating Daily Averages

```python
import pandas as pd

# Example DataFrame with a date column
data = {
    'timestamp': [
        '2023-01-01 08:00:00',
        '2023-01-01 10:00:00',
        '2023-01-01 15:00:00',
        '2023-01-02 09:00:00',
        '2023-01-02 18:00:00'
    ]
}
df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Group by date
df['date'] = df['timestamp'].dt.date

# Method 1: Average using (start + end) / 2
averages_method_1 = df.groupby('date')['timestamp'].agg(lambda x: (x.min() + x.max()) / 2)
# Correct Method 1: Convert timestamps to integers (UNIX timestamps), perform operations, then convert back
averages_method_1 = df.groupby('date')['timestamp'].agg(lambda x: pd.Timestamp((x.min().timestamp() + x.max().timestamp()) / 2, unit='s'))

# Method 2: Average of all timestamps in a day
averages_method_2 = df.groupby('date')['timestamp'].mean()

print("Method 1 Averages:")
print(averages_method_1)

print("\nMethod 2 Averages:")
print(averages_method_2)
```

---
