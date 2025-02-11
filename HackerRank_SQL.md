## 1. Airplane Bookings

You are provided with data of airplane bookings which contain total seats in an airplane 
and the bookings done.
Every airplane has some seats that are not booked.
Find out the average number of seats that go without booking for every airline 
and fetch the airplanes for each airline whose number of empty seats is closest to
the average number of seats that remain empty.

```sql
WITH empty_seats_per_airplane AS (
    -- define the empty seats per airplane
    SELECT
        a.airplane_id,
        a.airline_id,
        a.total_seats - COALESCE(SUM(b.booked)) AS empty_seats
    FROM airlines_detail a
    INNER JOIN bookings b ON a.airplane_id = b.airplane_id
    GROUP BY airplane_id, airline_id, total_seats
),

empty_seats_per_airline AS (
    -- define the empty seats per airline
    SELECT 
        airline_id,
        AVG(empty_seats) AS avg_empty_seats
    FROM empty_seats_per_airplane
    GROUP BY airline_id
),

closest_empty AS (
    -- find the airplane(s) with empty seats closest to the airline’s average
    SELECT 
        c.airplane_id,
        c.airline_id,
        c.empty_seats,
        ABS(c.empty_seats - d.avg_empty_seats) AS diff
    FROM empty_seats_per_airplane c
    INNER JOIN empty_seats_per_airline d ON c.airline_id = d.airline_id
)

SELECT 
    c.airline_id,
    GROUP_CONCAT(c.airplane_id ORDER BY c.airplane_id ASC SEPARATOR ',') AS airplane_ids
FROM closest_empty c
INNER JOIN (
    -- Find the minimum diff for each airline
    SELECT
        airline_id,
        MIN(diff) AS min_diff
    FROM closest_empty
    GROUP BY airline_id
) m ON c.airline_id = m.airline_id AND c.diff = m.min_diff
GROUP BY c.airline_id
ORDER BY c.airline_id;
```

---

## 2. SQL: Visitors Behavior Report 3

To develop visitor tracking software that lists the lengths of the longest sequence of events by day of the week in May 2022, generate a report with the following structure:

### Report Structure

- **type**: Represents the event type.  
- **Sunday .. Saturday**: Columns for each day of the week, showing the longest sequence length of a specific event type on that day in May 2022.

```sql
WITH filtered_events AS (
    -- Step 1: Filter only events from May 2022 and extract weekday
    SELECT 
        type, 
        dt, 
        DAYNAME(dt) AS weekday,
        LAG(dt) OVER (PARTITION BY type ORDER BY dt) AS prev_dt
    FROM events
    WHERE dt BETWEEN '2022-05-01' AND '2022-05-31'
),
grouped_events AS (
    -- Step 2: Identify if the event starts a new sequence (gap > 3 hours)
    SELECT 
        type, 
        dt, 
        weekday,
        CASE 
            WHEN prev_dt IS NULL OR TIMESTAMPDIFF(HOUR, prev_dt, dt) > 3 THEN 1 
            ELSE 0 
        END AS new_sequence
    FROM filtered_events
),
sequence_identification AS (
    -- Step 3: Assign a unique sequence group ID
    SELECT 
        type, 
        dt, 
        weekday,
        SUM(new_sequence) OVER (PARTITION BY type ORDER BY dt) AS group_id
    FROM grouped_events
),
sequence_counts AS (
    -- Step 4: Count events in each sequence and only keep valid ones (at least 2 events)
    SELECT 
        type, 
        weekday, 
        group_id, 
        COUNT(*) AS sequence_length
    FROM sequence_identification
    GROUP BY type, weekday, group_id
    HAVING COUNT(*) >= 2
),
max_sequences AS (
    -- Step 5: Find the longest sequence per type per weekday
    SELECT 
        type, 
        weekday, 
        MAX(sequence_length) AS max_sequence
    FROM sequence_counts
    GROUP BY type, weekday
)
-- Step 6: Pivot results to show weekdays as separate columns (Sunday-Saturday)
SELECT 
    type,
    COALESCE(MAX(CASE WHEN weekday = 'Sunday' THEN max_sequence END), NULL) AS Sunday,
    COALESCE(MAX(CASE WHEN weekday = 'Monday' THEN max_sequence END), NULL) AS Monday,
    COALESCE(MAX(CASE WHEN weekday = 'Tuesday' THEN max_sequence END), NULL) AS Tuesday,
    COALESCE(MAX(CASE WHEN weekday = 'Wednesday' THEN max_sequence END), NULL) AS Wednesday,
    COALESCE(MAX(CASE WHEN weekday = 'Thursday' THEN max_sequence END), NULL) AS Thursday,
    COALESCE(MAX(CASE WHEN weekday = 'Friday' THEN max_sequence END), NULL) AS Friday,
    COALESCE(MAX(CASE WHEN weekday = 'Saturday' THEN max_sequence END), NULL) AS Saturday
FROM max_sequences
GROUP BY type
ORDER BY type;
```

---

## 3. Advertising System Net Seller Report

```sql
WITH ordered_events AS (
    SELECT 
        e.campaign_id,
        e.type,
        e.amount,
        c.name AS campaign_name,
        ROW_NUMBER() OVER (PARTITION BY e.campaign_id ORDER BY (SELECT NULL)) AS row_num
    FROM events e
    JOIN campaigns c ON e.campaign_id = c.id
),
sequence_flags AS (
    SELECT 
        *,
        LAG(type) OVER (PARTITION BY campaign_id ORDER BY row_num) AS prev_type,
        CASE 
            WHEN LAG(type) OVER (PARTITION BY campaign_id ORDER BY row_num) <> type OR 
                 LAG(type) OVER (PARTITION BY campaign_id ORDER BY row_num) IS NULL 
            THEN 1 ELSE 0 
        END AS new_sequence
    FROM ordered_events
),
sequence_groups AS (
    SELECT 
        *,
        SUM(new_sequence) OVER (PARTITION BY campaign_id ORDER BY row_num) AS sequence_id
    FROM sequence_flags
),
sequence_summary AS (
    SELECT 
        campaign_name,
        campaign_id,
        sequence_id,
        COUNT(CASE WHEN type = 'sell' THEN 1 END) AS sell_count,
        COUNT(CASE WHEN type = 'buy' THEN 1 END) AS buy_count,
        SUM(CASE WHEN type = 'sell' THEN amount ELSE 0 END) AS sell_total
    FROM sequence_groups
    GROUP BY campaign_name, campaign_id, sequence_id
),
valid_sequences AS (
    SELECT 
        campaign_name,
        campaign_id,
        sequence_id,
        sell_total
    FROM sequence_summary
    WHERE sell_count > buy_count
)
SELECT 
    campaign_name AS campaign,
    COUNT(DISTINCT sequence_id) AS netsells_count,
    ROUND(SUM(sell_total), 2) AS netsells_total
FROM valid_sequences
GROUP BY campaign_name
ORDER BY netsells_total DESC;
