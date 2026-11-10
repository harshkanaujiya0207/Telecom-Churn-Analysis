"""
Synthetic Indian telecom churn dataset generator (10,000 rows).

DATA IS SIMULATED. Churn patterns are built in by this script's logic,
not observed from real customers. State this clearly in your README.

Run:  python generate_dataset.py
Out:  data/india_telecom_churn_synthetic.csv
"""

import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 10_000

# ---------- Customer profile ----------
operator = rng.choice(["Jio", "Airtel", "Vi", "BSNL"], N, p=[0.40, 0.33, 0.17, 0.10])
plan_type = rng.choice(["Prepaid", "Postpaid"], N, p=[0.82, 0.18])
age = np.clip(rng.normal(34, 11, N), 18, 75).astype(int)
gender = rng.choice(["Male", "Female"], N, p=[0.58, 0.42])

circles = ["Maharashtra", "Delhi", "Uttar Pradesh", "Karnataka", "Tamil Nadu",
           "West Bengal", "Gujarat", "Rajasthan", "Bihar", "Telangana"]
circle = rng.choice(circles, N, p=[0.14, 0.10, 0.18, 0.09, 0.09, 0.08, 0.08, 0.08, 0.08, 0.08])
area_type = rng.choice(["Urban", "Semi-urban", "Rural"], N, p=[0.45, 0.30, 0.25])

tenure_months = np.clip(rng.gamma(2.2, 14, N), 1, 120).astype(int)

# ---------- Plan & usage ----------
base_recharge = np.where(plan_type == "Prepaid",
                         rng.choice([199, 239, 299, 399, 479, 599, 719], N,
                                    p=[0.18, 0.17, 0.25, 0.15, 0.10, 0.10, 0.05]),
                         rng.choice([399, 499, 599, 799, 999, 1499], N,
                                    p=[0.18, 0.25, 0.22, 0.17, 0.12, 0.06]))
monthly_recharge = base_recharge
avg_data_gb = np.clip(rng.gamma(2.5, 5.5, N) * (monthly_recharge / 400) ** 0.5, 0.5, 120).round(1)
avg_call_minutes = np.clip(rng.normal(350, 160, N), 10, 1200).astype(int)
sms_sent = np.clip(rng.poisson(25, N), 0, 150)
device_type = rng.choice(["Android", "iPhone", "Feature phone"], N, p=[0.80, 0.10, 0.10])
is_5g = np.where(
    (device_type != "Feature phone") & (area_type != "Rural") & (operator != "BSNL"),
    rng.random(N) < 0.55, False)
payment_mode = rng.choice(["UPI", "Card", "Net banking", "Cash/Retailer"], N,
                          p=[0.55, 0.15, 0.10, 0.20])

# ---------- Service quality & behaviour ----------
op_network_risk = pd.Series(operator).map({"Jio": 0.9, "Airtel": 1.0, "Vi": 1.5, "BSNL": 1.8}).values
network_complaints = rng.poisson(0.6 * op_network_risk * np.where(area_type == "Rural", 1.5, 1.0))
customer_care_calls = rng.poisson(0.5 + 0.6 * network_complaints)
recharge_gap_days = np.clip(rng.gamma(2, 5, N) + 3 * network_complaints, 0, 90).astype(int)
num_recharges_6m = np.clip(rng.poisson(5, N) - (recharge_gap_days > 30) * 2, 0, 12)
last_complaint_resolved = np.where(
    network_complaints > 0, rng.choice(["Yes", "No"], N, p=[0.65, 0.35]), "No complaint")

# ---------- Churn logic (logistic) ----------
z = -2.0
z += 0.45 * network_complaints
z += 0.35 * (last_complaint_resolved == "No")
z += 0.04 * np.clip(recharge_gap_days - 15, 0, None)
z -= 0.30 * (num_recharges_6m - 5)
z -= 0.025 * np.clip(tenure_months, 0, 48)
z += 0.25 * (plan_type == "Prepaid")
z += 0.40 * (operator == "Vi") + 0.25 * (operator == "BSNL") - 0.20 * (operator == "Jio")
z -= 0.35 * is_5g
z += 0.003 * (400 - monthly_recharge) * (plan_type == "Prepaid") / 10
z += rng.normal(0, 0.6, N)  # noise so the signal is realistic, not perfect

churn_prob = 1 / (1 + np.exp(-z))
churn = (rng.random(N) < churn_prob).astype(int)

df = pd.DataFrame({
    "customer_id": [f"IN{100000 + i}" for i in range(N)],
    "operator": operator, "plan_type": plan_type, "age": age, "gender": gender,
    "circle": circle, "area_type": area_type, "tenure_months": tenure_months,
    "monthly_recharge_inr": monthly_recharge, "avg_data_gb": avg_data_gb,
    "avg_call_minutes": avg_call_minutes, "sms_sent": sms_sent,
    "device_type": device_type, "is_5g_user": is_5g.astype(int),
    "payment_mode": payment_mode, "network_complaints": network_complaints,
    "customer_care_calls": customer_care_calls,
    "complaint_resolved": last_complaint_resolved,
    "recharge_gap_days": recharge_gap_days, "num_recharges_6m": num_recharges_6m,
    "churn": churn,
})

# ---------- Light, realistic messiness (for the cleaning stage) ----------
for col, frac in [("avg_data_gb", 0.02), ("customer_care_calls", 0.015), ("age", 0.01)]:
    df.loc[rng.random(N) < frac, col] = np.nan
df.loc[rng.choice(N, 12, replace=False), "monthly_recharge_inr"] = 9999   # outliers
dups = df.sample(40, random_state=1)
df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=7).reset_index(drop=True)

os.makedirs("data", exist_ok=True)
out = "data/india_telecom_churn_synthetic.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows -> {out}")
print(f"Overall churn rate: {df['churn'].mean():.1%}")
