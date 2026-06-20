import duckdb
import pandas as pd
from pathlib import Path
import os

# =========================================================
# SETTINGS
# =========================================================

ROOT = Path(r".\dellstore_src\ds21-master\ds2\data_files")
OUT = Path("csv")

OUT.mkdir(exist_ok=True)

# =========================================================
# CONNECT
# =========================================================

con = duckdb.connect("dellstore.duckdb")

# =========================================================
# LOAD CUSTOMER TABLE
# =========================================================

print("\nLoading CUSTOMER table...")

cust_files = [
    ROOT / "cust" / "row_cust.csv",
    ROOT / "cust" / "us_cust.csv"
]

cust_df = pd.concat(
    [pd.read_csv(f) for f in cust_files],
    ignore_index=True
)

con.register("cust_df", cust_df)

con.execute("""
CREATE OR REPLACE TABLE CUSTOMER AS
SELECT * FROM cust_df
""")

print(f"CUSTOMER rows: {len(cust_df)}")

# =========================================================
# LOAD PRODUCTS
# =========================================================

print("\nLoading PRODUCTS table...")

prod_df = pd.read_csv(ROOT / "prod" / "prod.csv")

con.register("prod_df", prod_df)

con.execute("""
CREATE OR REPLACE TABLE PRODUCTS AS
SELECT * FROM prod_df
""")

print(f"PRODUCTS rows: {len(prod_df)}")

# =========================================================
# LOAD INVENTORY
# =========================================================

print("\nLoading INVENTORY table...")

inv_df = pd.read_csv(ROOT / "prod" / "inv.csv")

con.register("inv_df", inv_df)

con.execute("""
CREATE OR REPLACE TABLE INVENTORY AS
SELECT * FROM inv_df
""")

print(f"INVENTORY rows: {len(inv_df)}")

# =========================================================
# LOAD MONTHLY ORDER TABLES
# =========================================================

months = [
    "jan","feb","mar","apr",
    "may","jun","jul","aug",
    "sep","oct","nov","dec"
]

# ---------------- ORDERS ----------------

print("\nLoading ORDERS table...")

orders_df = pd.concat([
    pd.read_csv(ROOT / "orders" / f"{m}_orders.csv")
    for m in months
], ignore_index=True)

con.register("orders_df", orders_df)

con.execute("""
CREATE OR REPLACE TABLE ORDERS AS
SELECT * FROM orders_df
""")

print(f"ORDERS rows: {len(orders_df)}")

# ---------------- ORDERLINES ----------------

print("\nLoading ORDERLINES table...")

orderlines_df = pd.concat([
    pd.read_csv(ROOT / "orders" / f"{m}_orderlines.csv")
    for m in months
], ignore_index=True)

con.register("orderlines_df", orderlines_df)

con.execute("""
CREATE OR REPLACE TABLE ORDERLINES AS
SELECT * FROM orderlines_df
""")

print(f"ORDERLINES rows: {len(orderlines_df)}")

# ---------------- CUSTOMER HISTORY ----------------

print("\nLoading CUST_HIST table...")

custhist_df = pd.concat([
    pd.read_csv(ROOT / "orders" / f"{m}_cust_hist.csv")
    for m in months
], ignore_index=True)

con.register("custhist_df", custhist_df)

con.execute("""
CREATE OR REPLACE TABLE CUST_HIST AS
SELECT * FROM custhist_df
""")

print(f"CUST_HIST rows: {len(custhist_df)}")

# =========================================================
# EXPORT TABLES TO CSV
# =========================================================

print("\nExporting tables...\n")

tables = con.execute("SHOW TABLES").fetchall()

for (table,) in tables:

    out_file = OUT / f"{table}.csv"

    df = con.execute(f"SELECT * FROM {table}").df()

    df.to_csv(out_file, index=False)

    print(f"+ Exported {table}: {len(df)} rows")

# =========================================================
# SUMMARY
# =========================================================

print("\nTables created:")

for (table,) in tables:

    cnt = con.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]

    print(f" - {table}: {cnt} rows")

print("\nDone.")