# run this as a standalone script or in a notebook
# src/select_best_models.py

import sqlite3
import pandas as pd

ROOT     = "."
CLEAN_DB = "data/vayu_clean.db"

conn = sqlite3.connect(CLEAN_DB)

prophet = pd.read_sql("SELECT city, mae as prophet_mae FROM prophet_metrics", conn)
xgb     = pd.read_sql("SELECT city, mae as xgb_mae FROM xgb_regressor_metrics", conn)

merged = prophet.merge(xgb, on="city")
merged["best_model"] = merged.apply(
    lambda r: "prophet" if r["prophet_mae"] <= r["xgb_mae"] else "xgboost",
    axis=1
)
merged["best_mae"] = merged[["prophet_mae", "xgb_mae"]].min(axis=1).round(4)

merged.to_sql("forecast_model_selection", conn, if_exists="replace", index=False)
conn.close()

print(f"\n  {'City':<14} {'Prophet':>10}  {'XGBoost':>10}  {'Best':>10}  {'Best MAE':>10}")
print(f"  {'-'*55}")
for _, r in merged.sort_values("best_mae").iterrows():
    print(f"  {r['city']:<14} {r['prophet_mae']:>10.4f}  {r['xgb_mae']:>10.4f}  "
          f"{r['best_model']:>10}  {r['best_mae']:>10.4f}")
print(f"  {'-'*55}")
print(f"  {'Average':<14} {merged['prophet_mae'].mean():>10.4f}  "
      f"{merged['xgb_mae'].mean():>10.4f}  {'':>10}  "
      f"{merged['best_mae'].mean():>10.4f}")