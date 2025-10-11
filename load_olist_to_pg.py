# load_olist_to_pg.py
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

# connection settings (match docker-compose)
PG_USER = "ecom"
PG_PASS = "ecompass"
PG_DB   = "ecom_db"
PG_HOST = "localhost"     # from YOUR laptop to the container
PG_PORT = "5432"

DATA_DIR = Path("data/olist")  # folder with your CSVs

def smart_read(path: Path) -> pd.DataFrame:
    """
    Try a few encodings and separators so common CSV quirks don't block you.
    """
    encodings = ["utf-8", "utf-8-sig", "latin1"]
    seps = [",", ";", "\t", "|"]
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine="python", on_bad_lines="skip")
                # consider it valid if at least 2 columns read
                if df.shape[1] >= 2:
                    return df
            except Exception:
                continue
    raise RuntimeError(f"Could not parse {path}")

def main():
    assert DATA_DIR.exists(), f"Data folder not found: {DATA_DIR}"
    csvs = sorted([p for p in DATA_DIR.iterdir() if p.suffix.lower() in (".csv", ".txt")])
    assert csvs, f"No CSV files found in {DATA_DIR}"

    engine = create_engine(f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}")

    for f in csvs:
        print(f"Reading {f.name} ...")
        df = smart_read(f)

        # clean column names a bit: lowercase, underscores
        df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

        table = f.stem.lower()   # e.g., olist_orders_dataset
        print(f"Writing {len(df)} rows to table '{table}'")
        df.to_sql(table, engine, if_exists="replace", index=False, method="multi", chunksize=5000)

    print("✅ Done loading all files.")

if __name__ == "__main__":
    main()
