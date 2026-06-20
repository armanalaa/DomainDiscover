"""
extract_sql_to_csv.py
=====================
Extracts table data from a MySQL SQL dump file into individual CSV files.
Works without MySQL installed — pure Python SQL INSERT parser.

Handles:
  - Standard INSERT INTO `table` VALUES (...)
  - Multi-row INSERT INTO `table` VALUES (...),(...)
  - Escaped strings, NULL values, quoted identifiers

Usage:
    python extract_sql_to_csv.py --sql sakila-data.sql --schema sakila-schema.sql --out_dir csv
    python extract_sql_to_csv.py --sql dump.sql --out_dir csv
"""

import argparse
import csv
import re
import os
from pathlib import Path


# =============================================================================
# Column name extraction from schema file
# =============================================================================

def extract_columns_from_schema(schema_path: Path) -> dict:
    """
    Parse CREATE TABLE statements to get column names per table.
    Returns { table_name: [col1, col2, ...] }
    """
    if not schema_path or not schema_path.exists():
        return {}

    text = schema_path.read_text(encoding="utf-8", errors="replace")
    tables = {}

    # Find all CREATE TABLE blocks
    pattern = re.compile(
        r"CREATE TABLE\s+[`'\"]?(\w+)[`'\"]?\s*\((.*?)\)\s*(?:ENGINE|;)",
        re.DOTALL | re.IGNORECASE
    )

    for match in pattern.finditer(text):
        table_name = match.group(1)
        body       = match.group(2)
        cols = []
        for line in body.splitlines():
            line = line.strip().rstrip(",")
            # Skip constraints and keys
            if re.match(r"(PRIMARY|UNIQUE|KEY|CONSTRAINT|INDEX|FULLTEXT)", line, re.IGNORECASE):
                continue
            # Extract column name — first backtick-quoted or bare word
            m = re.match(r"[`'\"]?(\w+)[`'\"]?\s+\w", line)
            if m:
                cols.append(m.group(1))
        if cols:
            tables[table_name] = cols
            print(f"  [schema] {table_name}: {len(cols)} columns")

    return tables


# =============================================================================
# Value parser — handles MySQL-escaped strings
# =============================================================================

def parse_values_line(line: str) -> list[list]:
    """
    Parse VALUES (v1,v2,...),(v1,v2,...) into list of rows.
    Handles: NULL, integers, floats, quoted strings with escapes.
    """
    rows = []
    i    = 0
    n    = len(line)

    while i < n:
        # Find opening paren
        while i < n and line[i] != '(':
            i += 1
        if i >= n:
            break
        i += 1  # skip '('

        row    = []
        val    = []
        in_str = False
        quote  = None

        while i < n:
            c = line[i]

            if in_str:
                if c == '\\' and i + 1 < n:
                    nc = line[i + 1]
                    escape_map = {'n': '\n', 't': '\t', 'r': '\r',
                                  '\\': '\\', "'": "'", '"': '"', '0': '\x00'}
                    val.append(escape_map.get(nc, nc))
                    i += 2
                    continue
                elif c == quote:
                    in_str = False
                    i += 1
                    continue
                else:
                    val.append(c)
            else:
                if c in ("'", '"'):
                    in_str = True
                    quote  = c
                elif c == ',':
                    token = "".join(val).strip()
                    row.append(None if token.upper() == "NULL" else token)
                    val = []
                elif c == ')':
                    token = "".join(val).strip()
                    row.append(None if token.upper() == "NULL" else token)
                    rows.append(row)
                    i += 1
                    break
                else:
                    val.append(c)
            i += 1

    return rows


# =============================================================================
# Main extractor
# =============================================================================

def extract_sql_to_csv(
    sql_path:    Path,
    schema_path: Path,
    out_dir:     Path,
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load column definitions from schema file
    col_map = extract_columns_from_schema(schema_path) if schema_path else {}

    print(f"\n[parse] Reading {sql_path.name} ...")
    text = sql_path.read_text(encoding="utf-8", errors="replace")

    # Find all INSERT INTO statements — two passes:
    # Pass 1: INSERT INTO `table` (col1, col2) VALUES (...)  — with column list
    # Pass 2: INSERT INTO `table` VALUES (...)               — without column list
    insert_with_cols = re.compile(
        r"INSERT INTO\s+[`'\"]?(\w+)[`'\"]?\s*"
        r"\(([^)]+)\)\s*"
        r"VALUES\s*(.*?);",
        re.DOTALL | re.IGNORECASE
    )
    insert_no_cols = re.compile(
        r"INSERT INTO\s+[`'\"]?(\w+)[`'\"]?\s*"
        r"VALUES\s*(.*?);",
        re.DOTALL | re.IGNORECASE
    )

    # Collect all rows per table
    table_data:    dict[str, list] = {}
    table_columns: dict[str, list] = {}
    matched_spans: set             = set()

    # Pass 1 — inserts with explicit column list
    for match in insert_with_cols.finditer(text):
        table_name  = match.group(1)
        col_list    = match.group(2)
        values_text = match.group(3).strip()
        matched_spans.add(match.start())

        cols = [c.strip().strip("`'\"") for c in col_list.split(",")]
        rows = parse_values_line(values_text)

        if table_name not in table_data:
            table_data[table_name]    = []
            table_columns[table_name] = cols
        table_data[table_name].extend(rows)

    # Pass 2 — inserts without column list (use schema or auto-generate)
    for match in insert_no_cols.finditer(text):
        if match.start() in matched_spans:
            continue   # already handled in pass 1
        table_name  = match.group(1)
        values_text = match.group(2).strip()

        cols = col_map.get(table_name, None)
        rows = parse_values_line(values_text)

        if table_name not in table_data:
            table_data[table_name]    = []
            table_columns[table_name] = cols
        table_data[table_name].extend(rows)

        # Infer column names from row width if still unknown
        if table_columns[table_name] is None and rows:
            table_columns[table_name] = [f"col_{i+1}" for i in range(len(rows[0]))]

    # Write CSVs
    print(f"\n[write] Writing {len(table_data)} tables to {out_dir}\n")
    print("=" * 60)

    for table_name, rows in sorted(table_data.items()):
        cols = table_columns[table_name] or [f"col_{i+1}" for i in range(len(rows[0]) if rows else 0)]
        out_path = out_dir / f"{table_name}.csv"

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            writer.writerows(rows)

        print(f"  [OK] {table_name:<30} {len(rows):>6} rows  →  {out_path.name}")

    print("=" * 60)
    print(f"Done — {len(table_data)} tables extracted to {out_dir}")
    print("=" * 60)


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract MySQL SQL dump to CSV files (no MySQL needed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sakila with schema for column names
  python extract_sql_to_csv.py --sql sakila-data.sql --schema sakila-schema.sql --out_dir csv

  # Any dump without schema (column names auto-generated as col_1, col_2 ...)
  python extract_sql_to_csv.py --sql dump.sql --out_dir csv
"""
    )
    parser.add_argument("--sql",    required=True,
                        help="Path to the SQL data dump file (INSERT statements)")
    parser.add_argument("--schema", default=None,
                        help="Path to the SQL schema file (CREATE TABLE statements). "
                             "Used to get proper column names. Optional but recommended.")
    parser.add_argument("--out_dir", default="csv",
                        help="Output folder for CSV files (default: csv)")
    args = parser.parse_args()

    sql_path    = Path(args.sql)
    schema_path = Path(args.schema) if args.schema else None
    out_dir     = Path(args.out_dir)

    if not sql_path.exists():
        print(f"[ERROR] SQL file not found: {sql_path}")
        return

    print("=" * 60)
    print("SQL Dump → CSV Extractor")
    print(f"  Data file   : {sql_path}")
    print(f"  Schema file : {schema_path or '(none — column names auto-generated)'}")
    print(f"  Output dir  : {out_dir}")
    print("=" * 60)

    extract_sql_to_csv(sql_path, schema_path, out_dir)


if __name__ == "__main__":
    main()