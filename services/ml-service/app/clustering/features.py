import pandas as pd


def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    # Proporsi pengeluaran per kategori -- ini sinyal paling kuat untuk membedakan
    # segmen (misal user "investor" harusnya proporsi investasi-nya tinggi)
    category_pivot = (
        df.pivot_table(index="user_id", columns="category", values="amount", aggfunc="sum", fill_value=0)
    )
    category_proportion = category_pivot.div(category_pivot.sum(axis=1), axis=0)
    category_proportion.columns = [f"pct_{c}" for c in category_proportion.columns]

    # Fitur perilaku tambahan di luar proporsi kategori
    behavior = df.groupby("user_id").agg(
        avg_transaction_amount=("amount", "mean"),
        total_transactions=("transaction_id", "count"),
        total_spending=("amount", "sum"),
        segment_ground_truth=("segment_ground_truth", "first"),  # sama utk semua baris user ini
    )

    features = category_proportion.join(behavior)
    return features