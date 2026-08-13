import pandas as pd


def build_daily_expense_series(transactions_path: str, exclude_categories: list[str] | None = None) -> pd.DataFrame:
    exclude_categories = exclude_categories or ["gaji_pendapatan"]

    df = pd.read_csv(transactions_path)
    df = df[~df["category"].isin(exclude_categories)]
    df["date"] = pd.to_datetime(df["date"])

    daily = (
        df.groupby(["user_id", "date"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "daily_expense"})
    )

    # Isi hari-hari tanpa transaksi dengan 0 -- deret waktu harus lengkap/kontinu
    # tanpa lubang tanggal, atau model time series akan salah interpretasi jarak waktu.
    filled_series = []
    for user_id, group in daily.groupby("user_id"):
        date_range = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        group = group.set_index("date").reindex(date_range, fill_value=0)
        group["user_id"] = user_id
        group.index.name = "date"
        filled_series.append(group.reset_index())

    return pd.concat(filled_series, ignore_index=True)