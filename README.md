# Telco Customer Churn Prediction

Predicting which customers are likely to cancel their subscription, using machine learning on real-world telecom dataset.

## My Motivation

I work at ACE Pickleball Club, and recently we've seen a number of members cancel their memberships. That raised the question "what actually drives someone to leave a subscription-based business?", and could the likeliness be predicted in advance? 

To explore this, I used the publicly available Telco Customer Churn dataset(a well-documented dataset from a telecom company) to build and validate a churn prediction pipeline. 

The techniques and insights here (contract flexibility, pricing, and tenure driving churn) generalize well beyond telecom, and are directly relevant to any subscription or membership-based business, like gyms, clubs, streaming services and much more.

## Dataset

- **Source:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 features
- **Target:** `Churn` (Yes/No) — whether the customer left within the last month

## My Steps

1. **Cleaning the Data** — Identified and handled 11 rows with missing `TotalCharges` values (all new customers with 0 months of tenure, so no billing history existed yet). Dropped these rows (which was about 0.15% of data). 

2. **Exploratory Data Analysis** — Investigated relationships between churn and key features before modeling, to build intuition and identify possible predictive patterns.

3. **Feature engineering** — One-hot encoded all categorical variables (contract type, internet service, payment method, etc.), producing 30 model-ready features.

4. **Modeling** — Trained a Logistic Regression classifier on an 80% train and 20% test split, using standardized (scaled) features.

5. **Evaluation** — Assessed the model using accuracy, precision, and recall, with particular attention to recall on the churn class.

## My Key Findings

- **Contract type is a massive driver of churn.** Monthly customers churn at **42.7%**, compared to just **2.8%** for two-year contracts, which is a 15x difference.

- **Tenure matters.** Customers who churned had an average tenure of 18 months, vs. 37.6 months for those who stayed.

- **Effects of Pricing** Churned customers paid ~$13/month more on average than those who stayed.

- **Fiber optic customers churn more** (41.9%) than DSL (19.0%) or no-internet customers (7.4%), likely tied to the higher price point of fiber plans.

**Key Takeaway** Customers on flexible, high-cost, no-commitment plans are the highest of churn risk. A business could proactively target this population with retention incentives, such as discounts for switching to longer-term contracts in order to sustain their clients.

## Model Performance

| Metric | Class 0 (No Churn) | Class 1 (Churn) |
|---|---|---|
| Precision | 0.85 | 0.65 |
| Recall | 0.89 | 0.57 |
| F1-score | 0.87 | 0.61 |

**Overall accuracy: 80.4%** (vs. a 73.5% baseline from always predicting "no churn")

The model performs solidly overall but has room to improve on catching actual churners (57% recall) — likely due to class imbalance (only ~27% of customers churned in this dataset).

## How to Run This Project

1. Clone this repository: https://github.com/Ashwika-R/customer-churn-prediction.git

2. Create a conda environment: 
```
conda env create -f environment.yml
conda activate churn-project
```

3. Open `01_explore_data.ipynb` in VS Code or Jupyter and run all cells.

## Next Steps

- [ ] Address class imbalance to improve recall on churn class (e.g., SMOTE, class weighting)
- [ ] Compare against additional models (Random Forest, XGBoost)
- [ ] Feature importance analysis to rank the strongest churn predictors
- [ ] Build an interactive Streamlit demo for live predictions

## Tech Stack

Python · pandas · scikit-learn · matplotlib · seaborn · Jupyter
