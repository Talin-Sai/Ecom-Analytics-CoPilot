import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# --- DB connection (update credentials if needed)
engine = create_engine("postgresql://ecom:ecompass@localhost:5432/ecom_db")

# --- Pull data
sql = text("""
WITH orders AS (
  SELECT
    o.order_id,
    o.customer_id,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date
  FROM olist_orders_dataset o
  WHERE o.order_delivered_customer_date IS NOT NULL
    AND o.order_estimated_delivery_date IS NOT NULL
    AND o.order_purchase_timestamp IS NOT NULL
),
payments AS (
  SELECT
    p.order_id,
    SUM(p.payment_value)::numeric AS payment_value_total,
    MAX(p.payment_installments)    AS payment_installments_max
  FROM olist_order_payments_dataset p
  GROUP BY p.order_id
),
items AS (
  SELECT
    oi.order_id,
    SUM(oi.freight_value)::numeric AS freight_value_total
  FROM olist_order_items_dataset oi
  GROUP BY oi.order_id
)
SELECT
  o.order_id,
  o.customer_id,
  o.order_status,
  o.order_purchase_timestamp,
  o.order_approved_at,
  o.order_delivered_customer_date,
  o.order_estimated_delivery_date,
  COALESCE(py.payment_value_total, 0)    AS payment_value_total,
  COALESCE(py.payment_installments_max, 0) AS payment_installments_max,
  COALESCE(it.freight_value_total, 0)    AS freight_value_total
FROM orders o
LEFT JOIN payments py USING (order_id)
LEFT JOIN items    it USING (order_id)
""")

df = pd.read_sql(sql, engine)

# --- Feature engineering
to_dt = lambda s: pd.to_datetime(s, errors="coerce")
df["order_purchase_timestamp"]      = to_dt(df["order_purchase_timestamp"])
df["order_delivered_customer_date"] = to_dt(df["order_delivered_customer_date"])
df["order_estimated_delivery_date"] = to_dt(df["order_estimated_delivery_date"])

# Targets & features
df["actual_days"]   = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
df["estimated_days"] = (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.days
df["delayed"]        = (df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]).astype(int)

# Keep rows with valid features
df = df.dropna(subset=["actual_days", "estimated_days"])

# Fill numeric aggregates
for col in ["payment_value_total", "payment_installments_max", "freight_value_total"]:
    df[col] = df[col].fillna(0)

# target label
df["delayed"] = (df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]).astype(int)

# drop NAs
df = df.dropna(subset=["actual_days", "estimated_days", "delayed"])

print("Label distribution:\n", df["delayed"].value_counts())

# --- select features
features = ["actual_days", "estimated_days", "payment_value_total",
            "payment_installments_max", "freight_value_total"]
X = df[features]
y = df["delayed"]

print("Label distribution:\n", y.value_counts(dropna=False))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500))
])

# MLflow
import mlflow, mlflow.sklearn
mlflow.set_tracking_uri("http://localhost:5001")  # or 5000 if that’s yours
mlflow.set_experiment("ecom_delivery_delay")

with mlflow.start_run():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)
    mlflow.sklearn.log_model(pipe, "model")

    print(f"Accuracy: {acc:.3f}  F1: {f1:.3f}  AUC: {auc:.3f}")