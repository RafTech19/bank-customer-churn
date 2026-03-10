# bank-customer-churn
Customer Churn Analysis in the Banking Industry: Identifying Key Drivers and High-Risk Customer Segments

## Project Overview

Customer churn is one of the biggest challenges in the banking industry. Losing customers not only reduces revenue but also increases acquisition costs for new customers. This project aims to analyze customer churn behavior in a banking dataset and identify the key factors that contribute to customers leaving the bank. The insights from this analysis can help the bank design effective retention strategies and improve customer loyalty.

---

## Business Problem

Bank XYZ wants to reduce customer churn by understanding:

1. Which customer segments have the highest churn risk
2. What factors influence customer churn
3. How customer complaints affect churn
4. Whether customer activity level influences churn
5. How churn impacts the bank's financial performance

---

## Dataset Information
Kaggle : https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn
The dataset contains **10,000 customers** with **15 variables**, including:

- CustomerId
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary
- Exited (Target Variable)
- Complain
- SatisfactionScore
- CardType

---

## Data Cleaning Process

The following steps were performed:

- Checked for missing values (none found)
- Checked for duplicate customers
- Converted CustomerId from integer to string
- Created age group segmentation:
  - 18–30
  - 31–40
  - 41–50
  - 51+

---

## Exploratory Data Analysis

Several analyses were conducted to understand churn patterns:

### 1 Customer Demographics

Churn was analyzed based on:

- Gender
- Age Group
- Country

Key Findings:

- Female customers have higher churn rate (25.1%) than male customers (16.5%)
- Customers aged **41+** have the highest churn risk
- Germany has the highest churn rate (32.4%)

---

### 2 Product Ownership

Customers with **more than 2 products** show significantly higher churn risk.

Customers with **2 products have the lowest churn rate (~10%)**, indicating an optimal product engagement level.

---

### 3 Customer Tenure

- New customers (0–1 year) have higher churn risk due to low loyalty
- Long-term customers (7–10 years) show a slight increase in churn due to possible dissatisfaction or lack of new benefits

---

### 4 Customer Complaints

One of the strongest findings:

**99.5% of customers who churned had submitted complaints.**

This suggests complaint handling is a critical factor in customer retention.

---

### 5 Balance Impact

Customers who churned have **higher average balance**, meaning the bank is losing high-value customers.

---

### 6 Customer Activity Status

A **Chi-Square statistical test** shows a significant relationship between:

- Customer activity status
- Customer churn

Inactive customers are significantly more likely to leave the bank.

---

## Key Business Insights

The analysis reveals several high-risk segments:

- Female customers
- Customers aged 41+
- Customers located in Germany
- Customers with multiple products (>2)
- Customers who submitted complaints
- Inactive customers

---

## Business Recommendations

To reduce churn, the bank should focus on the following strategies:

### 1 High-Risk Segment Retention

Target retention strategies for:

- Female customers
- Older customers
- German market

---

### 2 Product Strategy Optimization

Simplify product offerings and focus on cross-selling up to **2 relevant products** per customer.

---

### 3 Complaint Handling Improvement

Improve customer service responsiveness and provide proactive compensation for service failures.

---

### 4 Customer Engagement Programs

Re-engage inactive customers using:

- Personalized notifications
- Loyalty programs
- Promotional offers

---

### 5 High-Balance Customer Protection

Create exclusive programs for **high-balance customers** such as:

- Premium services
- Wealth management products
- Priority customer support

---

## Dashboard

Interactive dashboard available on Tableau Public:

[View Dashboard](https://public.tableau.com/app/profile/muhammad.rafli.febriyanto/viz/P1M2_Muhamad_Rafli_Febriyanto/Dashboard1)

---
Presentation (Canva) : https://www.canva.com/design/DAHDFUTXE5U/b60Ix1glrk4pL5ZAStczKw/edit?utm_content=DAHDFUTXE5U&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton


## Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Tableau
- Jupyter Notebook

---

## Author

Muhammad Rafli Febriyanto

Data Analyst Enthusiast  
Email: rafli.bim05@gmail.com
