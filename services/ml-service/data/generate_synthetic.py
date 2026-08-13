import random
import sys
import string
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Tambahkan path root (/app) ke sys.path agar import "data.categories" dikenali
sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.categories import CATEGORIES, USER_SEGMENTS

random.seed(42) # reproducible

N_USERS = 200
MONTHS_OF_HISTORY = 6
ANOMALY_PROBABILITY = 0.02 # 2% transaksi sengaja dibuat anomali


def _pick_category(weights: dict) -> str:
    categories = list(weights.keys())
    probs = list(weights.values())
    return random.choices(categories, weights=probs, k=1)[0]


def _add_bank_statement_noise(merchant: str) -> str:
    style = random.choice(["qris", "transfer", "plain", "ecommerce"])

    if style == "qris":
        code = "".join(random.choices(string.digits, k=6))
        return f"QRIS-{merchant.upper()}-{code}"
    elif style == "transfer":
        date_code = f"{random.randint(1,28):02d}{random.randint(1,12):02d}"
        return f"TRSF E-BANKING/{merchant.upper().replace(' ', '')}/{date_code}"
    elif style == "ecommerce":
        return f"{merchant} - Payment #{random.randint(100000, 999999)}"
    else:
        return merchant


def generate_transaction(user_id: str, segment: str, date: datetime) -> dict:
    weights = USER_SEGMENTS[segment]["categories_weight"]
    category = _pick_category(weights)
    
    merchants = CATEGORIES[category]["merchants"]
    held_out = CATEGORIES[category].get("held_out_test", [])
    if held_out and random.random() < 0.15:  # 15% kesempatan menggunakan held-out merchant
        merchant = random.choice(held_out)
    else:
        merchant = random.choice(merchants)
        
    low, high = CATEGORIES[category]["amount_range"]
    amount = random.randint(low, high)
    
    is_anomaly = random.random() < ANOMALY_PROBABILITY
    if is_anomaly:
        # anomali: nominal jauh di luar kebiasaan (5-15x lipat normal)
        amount = int(amount * random.uniform(5, 15))

    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": user_id,
        "segment_ground_truth": segment,
        "date": date.strftime("%Y-%m-%d"),
        "merchant": merchant,
        "description": _add_bank_statement_noise(merchant),
        "category": category,
        "amount": amount,
        "is_anomaly_ground_truth": is_anomaly,
    }


def generate_dataset() -> pd.DataFrame:
    rows = []
    segments = list(USER_SEGMENTS.keys())

    for _ in range(N_USERS):
        user_id = str(uuid.uuid4())
        segment = random.choice(segments)
        low_tx, high_tx = USER_SEGMENTS[segment]["transactions_per_month"]

        start_date = datetime.now() - timedelta(days=30 * MONTHS_OF_HISTORY)

        for month in range(MONTHS_OF_HISTORY):
            n_tx = random.randint(low_tx, high_tx)
            for _ in range(n_tx):
                day_offset = month * 30 + random.randint(0, 29)
                tx_date = start_date + timedelta(days=day_offset)
                rows.append(generate_transaction(user_id, segment, tx_date))

    return pd.DataFrame(rows)


def main():
    df = generate_dataset()
    df.to_csv("shared_data/synthetic_transactions.csv", index=False)
    print(f"Generated {len(df)} transactions for {df['user_id'].nunique()} users")
    print(f"Anomalies injected: {df['is_anomaly_ground_truth'].sum()} ({df['is_anomaly_ground_truth'].mean():.2%})")
    print(df["category"].value_counts())


if __name__ == "__main__":
    main()