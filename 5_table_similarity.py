"""
=============================================================================
CCM Pipeline — Step 4: Aggregate to Table-Level Similarity
Follows Running_Example.docx §4.1 – §4.6
=============================================================================

STEPS IMPLEMENTED
────────────────────────────────────────────────────────────────────────────
§4.1  Load cross-column Sim_attr scores from Step 3 long CSV
§4.2  Greedy best-match per table pair:
        - Sort all cross-column pairs descending by Sim_attr
        - Assign each column at most once
        - Stop when the smaller table's columns are exhausted
§4.3  raw_sim = mean(selected match scores)
§4.4  Diagnostic: does normalization change the score? (CR < 1.0 flag)
§4.5  Normalize by coverage ratio:
        M_max = min(|Ti|, |Tj|)          §4.5.1
        CR    = |matches| / M_max        §4.5.2
        Sim_T = raw_sim × CR             §4.5.3
§4.6  Build Table Similarity Graph G_T:
        nodes  = tables
        edges  = (Ti, Tj, Sim_T)  only when Sim_T ≥ θ_T   §4.6.1–4.6.2

    M_max=min(4,2)=2,  CR=2/2=1.0
    Sim_T = 0.61 × 1.0 = 0.61
    0.61 > θ_T=0.60 → edge added  ✓

────────────────────────────────────────────────────────────────────────────
INPUTS  (produced by previous pipeline steps):
  step3_Sim_attr_long.csv   — all column pairs with Sim_attr  (column_graph.py)
  schema.json               — per-table column counts

OUTPUTS:
  step4_table_sim.csv       — all table pairs: raw_sim, M_max, CR, Sim_T
  step4_graph_edges.csv     — G_T edges above θ_T
  step4_report.txt          — human-readable summary

RUN:
  python table_similarity.py
  python table_similarity.py --input_dir ccm_output --theta_t 0.60
=============================================================================
"""

from __future__ import annotations
import os

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_INPUT_DIR  = Path("ccm_output")
DEFAULT_OUTPUT_DIR = Path("ccm_output")
DEFAULT_SCHEMA     = Path("schema.json")
DEFAULT_THETA_T    = 0.60


# =============================================================================
# §4.1 — Load cross-column similarities
# =============================================================================

def load_sim_attr(long_csv: Path) -> pd.DataFrame:
    """
    §4.1 — Load step3_Sim_attr_long.csv produced by column_graph.py.

    Expected columns:
        col_i, col_j, table_i, table_j, P_stat, P_stat_norm,
        P_name, P_sem, Sim_attr

    Each row is one column pair (upper triangle only, no self-pairs).
    """
    if not long_csv.exists():
        raise FileNotFoundError(
            f"Step 3 long CSV not found: {long_csv}\n"
            "Run column_graph.py first to produce step3_Sim_attr_long.csv"
        )
    df = pd.read_csv(long_csv)
    required = {"col_i", "col_j", "table_i", "table_j", "Sim_attr"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {long_csv.name}: {missing}")

    log.info("§4.1  Loaded %d column pairs from %s", len(df), long_csv.name)
    log.info("      Sim_attr range : [%.4f, %.4f]  mean=%.4f",
             df["Sim_attr"].min(), df["Sim_attr"].max(), df["Sim_attr"].mean())
    log.info("      Tables found   : %d",
             len(set(df["table_i"]) | set(df["table_j"])))
    return df


# =============================================================================
# Load column counts from schema.json
# =============================================================================

def load_col_counts(schema_path: Path, all_sim: pd.DataFrame) -> dict[str, int]:
    """
    Return {table_name: n_columns} used in §4.5.1 to compute M_max.

    Priority:
      1. schema.json  — exact counts from the schema definition
      2. Derived from the Sim_attr long CSV (fallback)
    """
    if schema_path.exists():
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        tables_data = schema.get("tables", {})
        counts: dict[str, int] = {}

        if isinstance(tables_data, dict):
            # Format: {"TableName": {"columns": [...], ...}, ...}
            for tbl_name, tbl_info in tables_data.items():
                cols = tbl_info.get("columns", [])
                counts[tbl_name] = len(cols)
        else:
            # Format: [{"name": "...", "columns": [...]}]
            for tbl in tables_data:
                counts[tbl["name"]] = len(tbl.get("columns", []))

        log.info("      Column counts loaded from %s (%d tables)",
                 schema_path.name, len(counts))
        return counts

    log.warning("schema.json not found at %s — deriving column counts from data",
                schema_path)
    cnt: Counter = Counter()
    for col in set(all_sim["col_i"]) | set(all_sim["col_j"]):
        tbl = col.rsplit(".", 1)[0] if "." in col else col
        cnt[tbl] += 1
    log.info("      Column counts derived (%d tables)", len(cnt))
    return dict(cnt)


# =============================================================================
# §4.2 — Greedy best-match for one table pair
# =============================================================================

def greedy_match(cross: pd.DataFrame) -> pd.DataFrame:
    """
    §4.2 — Greedy column matching for a single (Ti, Tj) pair.

    Algorithm (from Running Example §4.2):
      1. Sort all cross-column pairs by Sim_attr descending
      2. Pick the top pair; mark both columns as used
      3. Skip any pair that reuses an already-matched column
      4. Repeat until no valid pairs remain

    Returns DataFrame of selected (col_i, col_j, Sim_attr) rows.
    """
    sorted_pairs = cross.sort_values("Sim_attr", ascending=False).reset_index(drop=True)
    used: set[str] = set()
    selected = []

    for _, row in sorted_pairs.iterrows():
        ci, cj = row["col_i"], row["col_j"]
        if ci in used or cj in used:
            continue
        selected.append(row)
        used.add(ci)
        used.add(cj)

    return pd.DataFrame(selected).reset_index(drop=True)


# =============================================================================
# §4.3 + §4.4 + §4.5 — Aggregate and normalize one table pair
# =============================================================================

def aggregate_pair(
    ti: str,
    tj: str,
    all_sim: pd.DataFrame,
    col_counts: dict[str, int],
) -> dict:
    """
    Compute Sim_T for a single unordered table pair (Ti, Tj).

    §4.3  raw_sim = mean of greedy-selected match scores
    §4.4  norm_changed flag: True when CR < 1.0 (normalization reduces score)
    §4.5  Sim_T = raw_sim × CR
            M_max = min(|Ti|, |Tj|)      §4.5.1
            CR    = |matches| / M_max    §4.5.2
            Sim_T = raw_sim × CR         §4.5.3
    """
    # Collect all cross-column pairs for this table pair (both directions)
    mask = (
        ((all_sim["table_i"] == ti) & (all_sim["table_j"] == tj)) |
        ((all_sim["table_i"] == tj) & (all_sim["table_j"] == ti))
    )
    cross = all_sim[mask].copy()

    # Normalise direction: col_i always belongs to ti
    swap = cross["table_i"] == tj
    cross.loc[swap, ["col_i", "col_j"]] = \
        cross.loc[swap, ["col_j", "col_i"]].values
    cross.loc[swap, ["table_i", "table_j"]] = \
        cross.loc[swap, ["table_j", "table_i"]].values

    n_cols_i = col_counts.get(ti, 0)
    n_cols_j = col_counts.get(tj, 0)

    if len(cross) == 0:
        return {
            "table_i": ti, "table_j": tj,
            "n_cols_i": n_cols_i, "n_cols_j": n_cols_j,
            "n_cross_pairs": 0, "n_matches": 0,
            "raw_sim": 0.0, "M_max": 0, "CR": 0.0,
            "Sim_T": 0.0, "norm_changed": False,
        }

    # §4.2 greedy match
    matched   = greedy_match(cross[["col_i", "col_j", "Sim_attr"]])
    n_matches = len(matched)

    # §4.3 raw average
    raw_sim = float(matched["Sim_attr"].mean()) if n_matches > 0 else 0.0

    # §4.5.1 maximum possible matches
    M_max = min(n_cols_i, n_cols_j) if (n_cols_i > 0 and n_cols_j > 0) \
            else max(n_matches, 1)

    # §4.5.2 coverage ratio
    CR = n_matches / M_max if M_max > 0 else 0.0

    # §4.4 diagnostic flag
    norm_changed = abs(CR - 1.0) > 1e-6

    # §4.5.3 final normalized similarity
    Sim_T = raw_sim * CR

    return {
        "table_i":       ti,
        "table_j":       tj,
        "n_cols_i":      n_cols_i,
        "n_cols_j":      n_cols_j,
        "n_cross_pairs": len(cross),
        "n_matches":     n_matches,
        "raw_sim":       round(raw_sim, 6),
        "M_max":         M_max,
        "CR":            round(CR, 6),
        "Sim_T":         round(Sim_T, 6),
        "norm_changed":  norm_changed,
    }


# =============================================================================
# Compute all table pairs
# =============================================================================

def compute_all_pairs(
    all_sim: pd.DataFrame,
    col_counts: dict[str, int],
) -> pd.DataFrame:
    """
    Iterate over every unordered (Ti, Tj) pair and compute Sim_T.
    Returns a DataFrame sorted by Sim_T descending.
    """
    tables  = sorted(set(all_sim["table_i"]) | set(all_sim["table_j"]))
    n_pairs = len(tables) * (len(tables) - 1) // 2
    log.info("Computing Sim_T for %d tables → %d pairs ...", len(tables), n_pairs)

    t0 = time.perf_counter()
    results = [
        aggregate_pair(ti, tj, all_sim, col_counts)
        for i, ti in enumerate(tables)
        for j, tj in enumerate(tables)
        if j > i
    ]
    log.info("  Done — %d pairs in %.2f s", len(results), time.perf_counter() - t0)

    df = pd.DataFrame(results).sort_values("Sim_T", ascending=False).reset_index(drop=True)

    # §4.4 summary
    log.info("§4.4  Normalization (CR < 1.0) affected %d / %d pairs",
             int(df["norm_changed"].sum()), len(df))
    return df


# =============================================================================
# §4.6 — Build Table Similarity Graph G_T
# =============================================================================

def build_graph(table_sim: pd.DataFrame, theta_t: float) -> pd.DataFrame:
    """
    §4.6.1  Threshold θ_T controls graph density.
    §4.6.2  G_T = (V, E):
              V = tables
              E = {(Ti, Tj) | Sim_T ≥ θ_T}
              edge weight = Sim_T
    """
    edges = table_sim[table_sim["Sim_T"] >= theta_t][
        ["table_i", "table_j", "Sim_T"]
    ].copy().reset_index(drop=True)

    log.info("§4.6  G_T built: θ_T=%.2f → %d / %d pairs become edges (%.1f%%)",
             theta_t, len(edges), len(table_sim),
             100 * len(edges) / len(table_sim) if len(table_sim) else 0)
    return edges


# =============================================================================
# Running Example verification
# =============================================================================

# =============================================================================
# Verification
# =============================================================================

def running_example_check(table_sim: pd.DataFrame) -> None:
    """
    Log the top-3 most similar table pairs as a sanity check.
    No hardcoded table names — works for any dataset.
    """
    if table_sim.empty:
        log.warning("Table similarity check: no table pairs found.")
        return
    top = table_sim.nlargest(3, "Sim_T")
    log.info("Top-3 table similarity pairs (sanity check):")
    for _, r in top.iterrows():
        log.info(
            "  %s × %s  raw_sim=%.4f  CR=%.4f  Sim_T=%.4f",
            r["table_i"], r["table_j"], r["raw_sim"], r["CR"], r["Sim_T"]
        )


# =============================================================================
# Report
# =============================================================================

def write_report(
    table_sim: pd.DataFrame,
    graph_edges: pd.DataFrame,
    theta_t: float,
    path: Path,
) -> None:
    lines = [
        "=" * 70,
        "CCM Pipeline — Step 4 Report",
        "Aggregate to Table-Level Similarity",
        "=" * 70,
        f"Table pairs computed    : {len(table_sim)}",
        f"Threshold θ_T           : {theta_t}",
        f"Edges in G_T            : {len(graph_edges)}",
        f"Pairs where CR < 1.0    : {int(table_sim['norm_changed'].sum())}",
        "",
        "Sim_T distribution:",
        f"  min    : {table_sim['Sim_T'].min():.4f}",
        f"  max    : {table_sim['Sim_T'].max():.4f}",
        f"  mean   : {table_sim['Sim_T'].mean():.4f}",
        f"  median : {table_sim['Sim_T'].median():.4f}",
        "",
        f"  {'Ti':30s}  {'Tj':30s}  {'raw':>6}  "
        f"{'M_max':>5}  {'CR':>5}  {'Sim_T':>6}",
        "  " + "-" * 68,
    ]
    for _, r in table_sim.head(15).iterrows():
        lines.append(
            f"  {r['table_i']:30s}  {r['table_j']:30s}  "
            f"{r['raw_sim']:>6.4f}  {int(r['M_max']):>5d}  "
            f"{r['CR']:>5.3f}  {r['Sim_T']:>6.4f}"
        )
    lines += ["", f"G_T edges (Sim_T ≥ {theta_t}):",
              f"  {'Ti':30s}  {'Tj':30s}  {'Sim_T':>6}", "  " + "-" * 44]
    for _, r in graph_edges.iterrows():
        lines.append(
            f"  {r['table_i']:30s}  {r['table_j']:30s}  {r['Sim_T']:>6.4f}"
        )
    lines.append("=" * 70)
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    log.info("Report saved → %s", path)
    print("\n" + text)


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CCM Step 4 — Table-level similarity aggregation"
    )
    p.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR),
                   help="Directory with step3_Sim_attr_long.csv")
    p.add_argument("--schema",    default=str(DEFAULT_SCHEMA),
                   help="Path to schema.json")
    p.add_argument("--out_dir",   default=str(DEFAULT_OUTPUT_DIR),
                   help="Output directory")
    p.add_argument("--theta_t",   type=float, default=DEFAULT_THETA_T,
                   help=f"Table similarity threshold θ_T (default: {DEFAULT_THETA_T})")
    p.add_argument("--dataset_dir", default=None,
                   help="Dataset working directory containing schema.json, knowledge.docx, "
                        "csv/ and ccm_output/. When run_pipeline.py is in the parent folder, "
                        "pass the dataset subfolder name (e.g. --dataset_dir Chinook). "
                        "Defaults to the current working directory.")
    return p.parse_args()

    # ── Dataset directory — chdir so all relative paths resolve correctly ────
    if args.dataset_dir is not None:
        import os as _os
        _os.chdir(args.dataset_dir)


def main() -> None:
    args    = parse_args()
    in_dir  = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("CCM Pipeline — Step 4: Table-Level Similarity")
    log.info("  input_dir = %s", in_dir)
    log.info("  theta_T   = %.2f", args.theta_t)
    log.info("=" * 60)

    # §4.1 load
    all_sim    = load_sim_attr(in_dir / "step3_Sim_attr_long.csv")
    col_counts = load_col_counts(Path(args.schema), all_sim)

    # §4.2–§4.5 compute
    table_sim  = compute_all_pairs(all_sim, col_counts)
    running_example_check(table_sim)

    # §4.6 build graph
    graph_edges = build_graph(table_sim, args.theta_t)

    # Save
    table_sim.to_csv(out_dir / "step4_table_sim.csv",   index=False)
    graph_edges.to_csv(out_dir / "step4_graph_edges.csv", index=False)
    write_report(table_sim, graph_edges, args.theta_t, out_dir / "step4_report.txt")

    log.info("=" * 60)
    log.info("Step 4 complete.")
    log.info("  step4_table_sim.csv   → %s", out_dir / "step4_table_sim.csv")
    log.info("  step4_graph_edges.csv → %s", out_dir / "step4_graph_edges.csv")
    log.info("  Next: Step 5 — Graph Clustering (Louvain)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()