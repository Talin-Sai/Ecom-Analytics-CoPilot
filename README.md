# 🧠 Ecom-Analytics-CoPilot

**End-to-End Data Analytics & Machine Learning Platform**

---

## 📊 Overview
Ecom-Analytics-CoPilot is a full-stack e-commerce analytics project that demonstrates how raw transactional data can be transformed into actionable business insights and predictive intelligence using a modern data stack.

---

## ⚙️ Tech Stack
**Languages & Tools:**  
Python • SQL • Docker • PostgreSQL • dbt • Metabase • MLflow • GitHub Actions

---

## 🏗️ Architecture
1. **Data Ingestion:** Python ETL loads Olist CSV datasets into PostgreSQL.  
2. **Transformation:** dbt models clean and aggregate data (monthly sales, customer metrics).  
3. **Visualization:** Metabase dashboards display key performance trends.  
4. **Machine Learning:** Logistic Regression model predicts high-value customers, tracked with MLflow.  
5. **CI/CD:** GitHub Actions validate dbt builds and data tests.

---

## 🚀 Features
- Automated **Dockerized data infrastructure** (Postgres, Adminer, Metabase).  
- **Python ETL pipeline** for structured data ingestion.  
- **dbt transformations** with data quality checks and CI validation.  
- **Interactive Metabase dashboards** for sales and customer insights.  
- **MLflow-tracked model** achieving ROC-AUC = 0.84 for high-value customer prediction.  

---

## 🗂️ Project Structure
Ecom-Analytics-CoPilot/
│
├── configs/ # Environment and DB settings
├── ecom_dbt/ # dbt models and configs
├── scripts/ # ETL and ML scripts
├── assets/ # MLflow artifacts (plots, models)
├── notebooks/ # Exploratory analysis
├── docker-compose.yml # Container setup
└── requirements.txt
