# Customer Segmentation Analysis
**Advanced RFM-based Customer Clustering for Business Intelligence**

## Project Overview
This collaborative project implements dual methodologies for customer segmentation by integrating RFM analysis with K-Means clustering algorithms. Our team analyzed retail transaction data to identify distinct customer behavioral patterns. We explored two complementary strategies:

- **Approach A**: Utilizes RFM scoring methodology combined with K-means clustering algorithms
- **Approach B**: Employs raw RFM metrics directly with K-means clustering techniques

## Data Source
Our analysis leverages a comprehensive retail analytics dataset spanning three years of transaction records. The dataset was sourced from Kaggle's public repository and can be accessed [here](https://www.kaggle.com/kyanyoga/sample-sales-data).

## Project Rationale
Customer segmentation serves as a powerful analytical tool for understanding diverse client behaviors and tailoring services to meet varied customer requirements. Modern businesses accumulate vast amounts of transactional data, which can be strategically analyzed through segmentation techniques to develop targeted marketing strategies and boost revenue generation. RFM methodology stands out as one of the most effective approaches for creating customized promotional campaigns that drive sales growth.

**RFM Analysis** represents Recency, Frequency, and Monetary metrics - a proven framework for customer segmentation based on purchase behavior patterns. This analytical approach evaluates customers across three key dimensions:

- **Recency**: Time elapsed since the customer's most recent transaction
- **Frequency**: Total count of purchases made by each customer
- **Monetary**: Average spending amount per customer transaction

## Repository Structure
**Phase 1**: Primary methodology implementation - `Customer_segmentation.ipynb`
**Phase 2**: Alternative approach development - `Segmentation_Kmeans.ipynb`
**Automation**: Standalone segmentation script - `customers_segments.py`

## Technical Stack
Our implementation utilizes:
- Python 3.6 environment
- Essential libraries: pandas, numpy, sklearn, scipy, seaborn, matplotlib

## Execution Instructions
Launch the project using Jupyter Notebook or Google Colab platform.

For command-line execution:
