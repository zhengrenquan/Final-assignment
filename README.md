# Supermarket Sales Operations Dashboard
Streamlit Dashboard for Supermarket Chain Operations

## Step 1: Dataset & Decision-Maker
### Dataset
- Name: Supermarket Sales
- Source: Kaggle (https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales)
- Time Period: January 1, 2019 – March 31, 2019
- Size: 1000 transaction records across 3 branch cities
- Row Definition: One row represents one customer order, including branch city, product category, unit price, quantity, payment method, and customer rating.
- Data Type: Historical business transaction data

### Target User
Regional Operations Manager of the supermarket chain, responsible for daily performance, inventory restocking, and staff scheduling across 3 stores.

## Step 2: Dashboard Justification
### Usage Frequency
The operations manager views the dashboard every weekday morning, and conducts a weekly review every Monday.

### Decision Supported
If the operations manager sees a continuous drop in sales for a product category or an obviously below-average order value at a store on the dashboard, they will:
1. Adjust display positions or launch short-term promotions for the declining category
2. Visit the underperforming store to check service quality and adjust peak-hour staffing
3. Optimize next week's restocking plan based on sales trends to reduce overstock

### Current Pain Point
Without this dashboard, the manager has to export Excel files from store POS systems and manually calculate sales, order value, and category breakdowns. It usually takes 2–3 hours to get weekly figures. This delay causes stockouts to go unnoticed for up to a week, leading to lost revenue.

### Why a Dashboard (Not Email/Report/Alert)
Compared to weekly reports: The dashboard supports on-demand filtering by city and category, so managers can drill down into anomalies immediately.
Compared to alert systems: Daily operations focus on steady monitoring; frequent push notifications are unnecessary. A dashboard fits the habit of active daily review.

## Step 3: Dashboard Design
### Content Shown (Ordered by Importance)
1. **Top 3 headline KPIs**: Total Revenue, Total Orders, Average Order Value
    - Reason: The manager can judge overall business health within 5 seconds of opening the page.
2. **Daily sales trend chart**
    - Reason: Shows revenue fluctuations over time and quickly locates abnormal days.
3. **Category sales breakdown chart**
    - Reason: Compares category contribution and supports category management and restocking decisions.
4. **Sidebar filters**: City filter, product category filter
    - Reason: The most frequently used controls for single-store or single-category analysis.
5. **Collapsible detail table (at the bottom)**
    - Reason: Available for deep verification but does not distract from the main view.

### Two Items Intentionally Excluded
1. Customer gender / membership breakdown
    - Reason: The operations manager focuses on revenue and inventory turnover. Demographics are a marketing concern and do not change restocking or scheduling decisions.
2. Payment method distribution
    - Reason: Payment mix has no material impact on daily operations. Finance teams handle it separately.

### Data Limitations & Handling
- Only 3 months of data are available; year-over-year comparison is not possible. The data period is clearly labeled on the dashboard.
- Only 3 stores are covered; findings cannot be generalized to all stores nationwide.
- Some days have low order volume and may show high daily volatility; trend analysis is recommended on a weekly basis.

## Step 4: Refresh Frequency
### Data Refresh Cadence
In production, the data refreshes automatically every night, pulling full order data from the previous day's POS system.

### Review Cadence
The operations manager reviews once every weekday at 10 AM, plus a weekly review on Monday. Daily refresh is sufficient for decisions; real-time refresh would add system cost without changing any operational decisions.

### Action Trigger
When daily total revenue falls below 80% of the four-week average for the same weekday, an email alert is triggered so the manager can investigate the same day.

### Cost of Stale Data
If the data is one day out of date, daily sales anomalies are not caught promptly, which may lead to multi-day stockouts or staffing mismatches, causing an estimated 5–8% daily revenue loss per incident. A one-week lag would seriously degrade next-week restocking accuracy.

> Note: This dashboard uses a static Kaggle data snapshot. For production use, connect to a live POS data source.

## Project Files
- `app.py`: Streamlit dashboard source code
- `supermarket_sales.csv`: Raw transaction dataset from Kaggle
- `requirements.txt`: Python dependency list

## Deployed App Link
[[Put your streamlit app url here]](https://final-assignment-lrzjlxs8jsbmwj3pxsnjph.streamlit.app/)
