# ml/churn_quick_demo.py
import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import mlflow
import mlflow.sklearn

# 1) connect to your Postgres (from YOUR laptop)
engine = create_engine("postgresql://ecom:ecompass@localhost:5432/ecom_db")

# 2) build a SIMPLE customer features table from Olist
#    label = 1 if customer has > 1 orders (repeat buyer), else 0
sql = """
WITH orders AS (
  SELECT
    o.customer_id,
    (o.order_purchase_timestamp)::timestamp AS ts,
    o.order_id
  FROM olist_orders_dataset o
),
spend AS (
  SELECT
    o.customer_id,
    SUM(oi.price)::numeric AS total_spend,
    AVG(oi.price)::numeric AS avg_item_price
  FROM olist_order_items_dataset oi
  JOIN olist_orders_dataset o USING(order_id)
  GROUP BY o.customer_id
),
by_customer AS (
  SELECT
    customer_id,
    COUNT(*)::int AS num_orders,
    MIN(ts) AS first_order_ts,
    MAX(ts) AS last_order_ts
  FROM orders
  GROUP BY customer_id
),
features AS (
  SELECT
    bc.customer_id,
    bc.num_orders,
    EXTRACT(EPOCH FROM (bc.last_order_ts - bc.first_order_ts))/86400.0 AS active_days,
    COALESCE(s.total_spend, 0)::float AS total_spend,
    COALESCE(s.avg_item_price, 0)::float AS avg_item_price
  FROM by_customer bc
  LEFT JOIN spend s ON s.customer_id = bc.customer_id
),
labeled AS (
  SELECT
    f.*,
    NTILE(5) OVER (ORDER BY f.total_spend DESC) AS spend_ntile
  FROM features f
)
SELECT
  customer_id,
  num_orders,
  active_days,
  total_spend,
  avg_item_price,
  CASE WHEN spend_ntile = 1 THEN 1 ELSE 0 END AS high_value
FROM labeled;
"""


df = pd.read_sql(text(sql), engine)

# 3) features / target
feature_cols = ["num_orders", "active_days", "total_spend", "avg_item_price"]
X = df[feature_cols].fillna(0.0)
y = df["high_value"].astype(int)
print("Label distribution:\n", y.value_counts(dropna=False))
if y.nunique() < 2:
    raise RuntimeError("Only one class present; check labeling SQL.")


# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 4) simple pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

# 5) MLflow tracking
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("ecom_high_value_buyer")

with mlflow.start_run(run_name="logreg_baseline"):
    pipe.fit(X_train, y_train)

    # predict
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # metrics
    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    # log
    mlflow.log_params({"model": "logreg", "features": ",".join(feature_cols)})
    mlflow.log_metrics({"auc": auc, "accuracy": acc, "f1": f1})

    # save model
    mlflow.sklearn.log_model(pipe, artifact_path="model")

print("Done. Check MLflow UI at http://localhost:5001")
