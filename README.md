# 🛒 Superstore Sales — End-to-End Exploratory Data Analysis (EDA)
### Python | Pandas | Matplotlib | Seaborn | Scikit-learn | Jupyter Notebook

> A comprehensive exploratory data analysis project on a retail Superstore dataset — covering data cleaning, feature engineering, customer segmentation, outlier handling, and multi-dimensional visual analysis across sales, profit, shipping, discounts, and regional performance.

---

## 📌 What This Project Is About

Retail data is rarely clean, rarely simple, and almost never tells its full story at first glance. This project takes a raw Superstore sales dataset and works through it systematically — fixing date inconsistencies, imputing missing values intelligently, engineering new business-relevant features, segmenting customers by profitability, and then building a comprehensive analytical dashboard that answers real business questions about product performance, regional profitability, discount impact, and shipping efficiency.

Every step is documented with the reasoning behind each decision — not just what was done, but why it was the right approach for this particular dataset.

---

## 📂 Dataset

**File:** `superstore_eda_v1.csv`
**Records:** ~9,997 rows (before cleaning)
**Source:** Retail Superstore transaction data

| Column | Description |
|---|---|
| `Order ID` | Unique order identifier (contains year in format) |
| `Order Date` | Date the order was placed |
| `Ship Date` | Date the order was shipped |
| `Ship Mode` | Shipping method — Same Day, Standard Class etc. |
| `Customer ID` | Unique customer identifier |
| `Customer Name` | Customer full name (dropped for PII) |
| `Segment` | Customer segment — Consumer, Corporate, Home Office |
| `Postal Code` | Delivery postal code |
| `City` | Delivery city |
| `State` | Delivery state |
| `Region` | Geographic region — West, East, Central, South |
| `Product Name` | Product description |
| `Category` | Product category — Technology, Furniture, Office Supplies |
| `Sub-Category` | Product sub-category |
| `Sales Price` | Unit price after discount |
| `Quantity` | Units ordered |
| `Discount` | Discount applied (0 to 1) |
| `Profit` | Profit per unit |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.x** | Core language |
| **Pandas** | Data loading, cleaning, transformation, pivot tables |
| **NumPy** | Numerical operations |
| **Matplotlib** | Bar charts, line plots, time series |
| **Seaborn** | Heatmaps, violin plots, scatter with regression, joint plots |
| **Scikit-learn** | Label encoding for categorical correlation analysis |
| **Jupyter Notebook** | Interactive analysis and documentation |

---

## 🔍 Project Walkthrough — Step by Step

---

### Step 1 — Data Loading and Initial Exploration
**What was done:**
Loaded the dataset into a pandas DataFrame and performed an initial structural inspection — checking data types via `.info()`, summary statistics via `.describe()`, and unique value distributions across all columns. Missing values were identified using `.isnull().sum()` and obvious inconsistencies were flagged before any cleaning began.

**Why it matters:**
You cannot clean data you haven't understood first. This step established the baseline picture of data quality issues that drove every subsequent decision.

---

### Step 2 — Handling Duplicates
**What was done:**
Identified and removed duplicate rows using `.drop_duplicates()`. Documented both the number of rows and distinct Order IDs affected.

**Key finding:**
7,459 rows were affected, impacting 2,471 unique Order IDs — a significant duplication rate that would have distorted every downstream aggregation without this step.

---

### Step 3 — Date Handling
**What was done:**
- Normalised `Order Date` and `Ship Date` using `pd.to_datetime()` with `format='mixed'` and `dayfirst=True` to handle inconsistent date formats across rows
- Extracted year from `Order ID` (which encodes the year in its format) and compared it against the year in `Order Date`
- Corrected mismatches by replacing the Order Date year with the Order ID year — the more reliable source
- Fixed cases where `Ship Date` was earlier than `Order Date` (negative Days to Ship) by correcting month/day swap errors in the raw data

**Key finding:**
40 records (0.40% of the dataset) had year inconsistencies between Order ID and Order Date. These were systematically corrected.

**Documented rationale:**
Fixing dates blindly can introduce new errors — root cause was identified first (month-day transposition in some rows) before applying corrections.

---

### Step 4 — Imputation of Missing Values
**What was done:**

**Ship Mode imputation:**
Calculated `Days to Ship` as `(Ship Date - Order Date).dt.days`. Used this to impute missing Ship Mode values:
- 0 days → "Same Day"
- 7 days → "Standard Class"
- Remaining nulls → "Standard Class" as the default fallback

**Quantity imputation:**
Used group-based median imputation — filling missing Quantity values with the median for that specific Product Name.

**Rationale for median:**
Quantity is a discrete variable that may contain outliers. The median is more robust than the mean for discrete data, and product-level grouping preserves realistic, context-appropriate estimates rather than applying a global average.

**Key finding:**
98 rows out of 9,997 were affected across all date and missing value operations combined.

---

### Step 5 — Data Masking and String Handling
**What was done:**
- Created `Customer Name Masked` — extracted initials from the full customer name using a lambda function that joins the first letter of each word
- Dropped the original `Customer Name` column to protect Personally Identifiable Information (PII)
- Converted `Postal Code` from numeric to text format using `.str.zfill(5)` to restore leading zeros lost during numeric storage

**Why it matters:**
Data protection regulations require that PII is not unnecessarily retained in analytical datasets. Masking retains a form of identification for analysis without exposing personal data.

---

### Step 6 — Data Type Conversion
**What was done:**
Converted `Quantity` from object to `int64` and `Sales Price` from object to `float64` to enable arithmetic operations that were blocked by incorrect string storage.

---

### Step 7 — Handling Inconsistent Categorical Data
**What was done:**
Cleaned the `State` column by replacing US state abbreviations with full state names using a mapping dictionary built from an external reference file (`State abbr. USA.xlsx`). Applied the mapping via a lambda function — retaining values unchanged if not found in the mapping. Also removed backslash characters from state entries that had been corrupted during data entry.

---

### Step 8 — Feature Engineering
**New columns created:**

| Feature | Formula | Purpose |
|---|---|---|
| `Original Price` | `Sales Price / (1 - Discount)` | Price before any discount |
| `Total Sales` | `Sales Price × Quantity` | Total revenue per order line |
| `Total Profit` | `Profit × Quantity` | Total profit per order line |
| `Discount Price` | `Original Price - Sales Price` | Actual discount amount in currency |
| `Total Discount` | `(Original Price × Discount) × Quantity` | Total discount value for quantity sold |
| `Days to Ship` | `(Ship Date - Order Date).dt.days` | Shipping duration in days |
| `Shipping Urgency` | Based on Days to Ship | Immediate (0d) / Urgent (1–3d) / Standard (>3d) |
| `Days Since Last Order` | `.diff().dt.days` per Customer ID | Customer purchase frequency metric |
| `Customer Total Sales` | Aggregated per Customer ID | Customer-level revenue contribution |
| `Customer Total Quantity` | Aggregated per Customer ID | Customer-level volume |
| `Customer Total Discount` | Aggregated per Customer ID | Customer-level discount received |

**Why feature engineering matters:**
Unit-level metrics (Sales Price, Profit) tell only part of the story. Total Sales and Total Profit at order line level are what actually drive business decisions. The Original Price calculation reversed the discount formula to reveal true pricing before promotions.

---

### Step 9 — Outlier Detection and Handling
**What was done:**

**Sales Price outliers (1.5×IQR):**
Identified outliers using the standard IQR method. Instead of removing them, capping was applied — values were clipped to the lower and upper bounds. This preserves dataset size while limiting extreme value influence.

**`remove_outliers()` function (3×IQR):**
Built a reusable function that accepts a DataFrame and column name, applies the 3×IQR rule, and returns the cleaned DataFrame.

**Why 3×IQR:**
The dataset has high variance in financial columns. The standard 1.5×IQR rule would flag too many legitimate high-value transactions as outliers. The 3×IQR method ensures only genuinely extreme values are removed — preserving business data integrity while still mitigating true outliers.

Also built `remove_outliers_multi()` as an extension that accepts multiple columns and applies the function sequentially.

---

### Step 10 — Customer Segmentation and Analysis
**What was done:**
- Aggregated `Total Sales` and `Total Profit` per Customer ID
- Created `Sales_Quintile` and `Profit_Quintile` using `pd.qcut()` dividing customers into 5 equal groups (Q1 = bottom 20%, Q5 = top 20%)
- Built a cross-tabulation (`pd.crosstab`) and heatmap comparing the two quintiles

**Key insight:**
High sales does not always translate into high profit. Some top-revenue customers fall into low profit quintiles — indicating inefficient pricing or aggressive discounting for those accounts.

---

### Step 11 — Final Analysis and Dashboard Creation

#### 📊 Sales and Profit Analysis
- **Top 10 Most Profitable Products** — bar chart showing highest total profit products
- **Top 10 Most Loss-Making Products** — bar chart showing products with largest negative profit (paired with average discount analysis)
- **Sales vs Profit Correlation** — scatter plot with regression line; correlation = **0.468** (moderate positive — higher sales generally increase profit but not reliably)
- **Joint Distribution** — Seaborn joint plot showing both the relationship and the spread between Total Sales and Total Profit

#### 👥 Customer Segmentation Analysis
- **Quintile Heatmap** — cross-tabulation of Sales Quintile vs Profit Quintile across all customer records
- **Category × Segment Pivot** — pivot table showing Total Sales and Total Profit broken down by product Category and customer Segment. Technology performs best; Furniture shows low profitability despite strong sales

#### 🚚 Shipping and Delivery Analysis
- **Shipping Urgency Distribution** — pie chart and bar chart showing order split: Standard (majority), Urgent, Immediate
- **Days to Ship vs Profit Violin Plot** — distribution of profit across Immediate / Fast / Slow shipping speeds
- **Shipping Mode Profitability** — bar chart comparing total profit by Ship Mode
- **Region × Ship Mode Pivot** — order count, total sales, and total profit for each Region and Ship Mode combination. Standard Class is the most preferred mode across all regions; Same Day shipping generates lower profit due to higher operational cost

#### 🗺️ Regional Sales and Profitability
- **Region-level bar charts** — Total Sales and Total Profit by region
- **State-wise Profitability Pivot** — sorted pivot table highlighting top and bottom 10 states by Total Profit
- **State vs Profit Correlation** — label-encoded State correlated with Total Profit (result: -0.016, confirming label encoding is not appropriate for categorical variables)

**Regional key findings:**
- **West** — top performer in both sales and profit
- **Central** — lowest profitability despite moderate sales (heavy discounting suspected)
- **South** — most efficient profit margin relative to sales volume
- **East** — high sales volume but proportionally lower profit

#### 💰 Discount and Pricing Analysis
- **Discount vs Profit scatter** — regression line confirms higher discounts reduce profit; some high-discount orders produce negative profit
- **Original vs Discounted Price line plot** — compares average original and discounted price across Categories. Technology maintains the highest post-discount value; Furniture and Office Supplies show the most aggressive discounting

#### 📅 Temporal Analysis
- **Monthly sales and profit time series** — identifies seasonal demand fluctuations
- **Order Frequency by Month** — bar chart highlighting peak and low order months
- **Year-over-Year Growth chart** — 2016 showed strong growth; 2017 saw profitability decline despite increasing sales, suggesting rising costs or discount escalation

---

## 📋 Key Business Insights Summary

| Area | Finding |
|---|---|
| Duplicates | 7,459 duplicate rows removed across 2,471 Order IDs |
| Date quality | 40 records had year mismatches between Order ID and Order Date |
| Sales–Profit correlation | Moderate at 0.468 — high sales ≠ guaranteed profit |
| Top region | West leads in both sales and profitability |
| Problem region | Central has lowest profit despite decent sales |
| Discount impact | Higher discounts consistently reduce and sometimes eliminate profit |
| Best category | Technology — highest profit and strongest pricing power post-discount |
| Worst category | Furniture — high sales but low profitability |
| Shipping | Standard Class dominates; Same Day increases cost and reduces margin |
| Customer insight | High-revenue customers are not always high-profit customers |

---

## 📁 Project Structure

```
Superstore-EDA/
│
├── Superstore.ipynb               # Main analysis notebook (152 cells)
├── superstore_eda_v1.csv          # Source dataset
├── State abbr. USA.xlsx           # US state abbreviation reference file
└── README.md                      # Project documentation
```

---

## ▶️ How to Run

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/Sky-paras/Superstore-EDA.git
cd Superstore-EDA
```

**Step 2 — Install dependencies:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl jupyter
```

**Step 3 — Update file paths:**
Open `Superstore.ipynb` and replace the local file paths at the top:
```python
# Replace this:
df = pd.read_csv(r"D:\Paras\My Document\...\superstore_eda_v1.csv")

# With this:
df = pd.read_csv("superstore_eda_v1.csv")
```

**Step 4 — Run the notebook:**
```bash
jupyter notebook Superstore.ipynb
```

---

## 🙌 Acknowledgements

- Dataset based on the widely used Superstore retail sample dataset
- Project completed as part of a Python Data Analysis and Statistics curriculum at **Coding Ninjas**

---

## 📬 Connect

Feel free to reach out to discuss the analysis, suggest improvements, or collaborate on a data project.

> ⭐ If this project was useful or interesting, a star on GitHub goes a long way — it helps others find it too.
