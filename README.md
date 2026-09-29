# 📊 HR Analytics: Predicting Employee Attrition & Retention Strategies

An end-to-end data analytics and machine learning case study designed to identify the primary drivers of employee turnover and predict flight risks using tree-based classification models.

## 🎯 Project Overview

The goal of this project is to equip HR leadership with actionable, data-driven insights to improve employee retention. By analyzing historical HR data, this project identifies critical attrition windows and workload thresholds. The final deliverable is a highly tuned Random Forest classifier optimized for *recall*, ensuring at-risk employees are reliably flagged for proactive intervention without generating excessive false alarms.

## 🧹 Dataset & Data Cleaning

The dataset contains employee performance metrics, workload hours, tenure, and satisfaction levels.

* **Initial shape:** 14,999 records.
* **Cleaning phase:** Dropped 3,008 statistically improbable duplicate rows to prevent severe data leakage and model bias, resulting in a clean dataset of **11,991 unique employee records**.

## 💡 Key Exploratory Data Analysis (EDA) Insights

* **⚖️ The Bimodal Workload Risk:** Employee churn risk is heavily bimodal. Extreme underutilization (2 projects) and severe burnout (6+ projects, 250+ hours) lead to disproportionately high turnover.
* **⏳ The 3-to-5-Year Itch:** Turnover spikes significantly between years 3 and 5, highlighting a critical window for targeted HR interventions.
* **❌ Misconceptions:** Department category and salary tier play almost no role in driving turnover compared to daily workload and overall job satisfaction.

## 🤖 Machine Learning Modeling

The prediction task is a binary classification problem (`left`: 0 or 1).

1. **Baseline Model (Logistic Regression):** Achieved 83% accuracy but suffered from a low recall of 21% on the minority class due to its inability to map the non-linear, bimodal churn triggers identified during EDA.
2. **Final Model (Tuned Random Forest):** Transitioned to a tree-based algorithm using `GridSearchCV` to specifically optimize for the `recall` metric. Applied balanced class weights to heavily penalize minority-class errors.

### 🏆 Final Model Performance (Random Forest)

* **Accuracy:** 98%
* **AUC Score:** 0.980
* **Precision:** 98%
* **Recall (Churned Employees):** 92%

By optimizing for recall, the Random Forest model successfully eliminated the false-negative issue, increasing the capture rate of at-risk employees from 21% to 92%.

## 📈 Top Attrition Drivers

Feature importance extraction reveals that the AI relies almost exclusively on three key variables to identify flight risks:

1. `satisfaction_level`
2. `tenure`
3. `number_project`

## 🤝 Business Recommendations

1. **Standardize Workload:** Restructure project allocations to target a sustainable baseline of 3 to 5 projects to prevent both extreme boredom and burnout.
2. **Targeted Check-ins:** Implement mandatory career progression interviews for employees entering their third year of tenure.
3. **Deployment:** Integrate the model alongside HR information systems to act as an early-warning radar, using the individual probability scores to prioritize retention efforts.

## 💻 Technologies Used

* **Python** (Pandas, NumPy)
* **Machine Learning** (Scikit-Learn, XGBoost)
* **Data Visualization** (Matplotlib, Seaborn)

## 📁 Repository Files

* **`data-driven-suggestions-for-hr-capstone-project.ipynb`**: The primary Jupyter Notebook containing the complete workflow, including exploratory data analysis (EDA), data cleaning, feature engineering, and the training and evaluation of the machine learning models.
* **`HR_comma_sep.csv`**: The raw HR dataset containing the 14,999 original employee records used for the analysis.
