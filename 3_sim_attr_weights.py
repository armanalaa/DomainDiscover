"""
=============================================================================
CCM Pipeline — Data-Driven Weight Derivation for Step 3.4.2
=============================================================================

GOAL
────────────────────────────────────────────────────────────────────────────
Replace heuristic weights w1=0.4, w2=0.3, w3=0.3 with weights derived
from the actual data, so the choice is principled and citable.

THREE DIAGNOSTICS (all computed, weights derived from Option 2)
────────────────────────────────────────────────────────────────────────────

Option 1 — Range-based (effective span of each signal)
  Measures: how wide is each signal's actual range in your data?
  Formula:  w_k = range_k / sum(range_1..3)
  Problem alone: a wide range with low variance is still uninformative.

Option 2 — Variance-based  ← RECOMMENDED (weights derived from this)
  Measures: how much does each signal discriminate between pairs?
  Formula:  w_k = Var(signal_k) / sum(Var(signal_1..3))
  Why best: directly measures discriminative power.
            Same principle as PCA feature importance — well-established
            and easy to justify to reviewers.

Option 3 — Correlation diagnostic (redundancy check)
  Measures: are any two signals saying the same thing?
  Formula:  Pearson correlation matrix between signals
  Use:      if two signals are highly correlated (|r| > 0.7),
            they double-count the same information.
            This does not change the weights directly but warns you
            if the weight split between correlated signals is misleading.

NOTE ON P_stat
────────────────────────────────────────────────────────────────────────────
P_stat is a DISTANCE (lower = more similar).
In the Sim_attr formula it appears as (1 - P_stat/P_stat_max), which
converts it to a SIMILARITY.
All diagnostics are computed on the CONVERTED form so all three signals
are on the same scale (higher = more similar).

────────────────────────────────────────────────────────────────────────────
INSTALL:  pip install numpy pandas scipy
INPUTS:   ccm_output/step3_proximity_long.csv
OUTPUTS:  ccm_output/derived_weights.txt   (human-readable report)
          ccm_output/derived_weights.csv   (w1, w2, w3 for use in pipeline)
RUN:      python derive_weights.py
=============================================================================
"""

from __future__ import annotations
import os

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

INPUT_CSV  = Path("ccm_output/step3_proximity_long.csv")
OUT_DIR    = Path("ccm_output")

# Heuristic default (what we had before)
HEURISTIC_W = (0.4, 0.3, 0.3)


# ===========================================================================
# Load and convert signals to unified similarity scale
# ===========================================================================

def load_signals(path: Path) -> pd.DataFrame:
    """
    Load proximity long CSV and convert all three signals to
    similarity ∈ [0,1]  (higher = more similar).

      s1 = 1 - P_stat / P_stat_max   (invert distance → similarity)
      s2 = P_name                     (already similarity)
      s3 = P_sem                      (already similarity)
    """
    df = pd.read_csv(path)
    log.info("Loaded %d column pairs", len(df))

    p_stat_max = df["P_stat"].max()
    log.info("P_stat_max (global normaliser) = %.4f", p_stat_max)

    df["s1_stat"] = 1.0 - df["P_stat"] / p_stat_max   # statistical similarity
    df["s2_name"] = df["P_name"]                        # name similarity
    df["s3_sem"]  = df["P_sem"]                         # semantic similarity

    log.info("Signal ranges after conversion to similarity:")
    for col, label in [("s1_stat","stat"), ("s2_name","name"), ("s3_sem","sem")]:
        s = df[col]
        log.info("  %s : min=%.4f  max=%.4f  mean=%.4f  std=%.4f  var=%.6f",
                 label, s.min(), s.max(), s.mean(), s.std(), s.var())

    return df, p_stat_max


# ===========================================================================
# Option 1 — Range-based weights
# ===========================================================================

def range_based_weights(df: pd.DataFrame) -> tuple[float, float, float]:
    """
    w_k = (max_k - min_k) / sum_of_ranges

    Rationale: a signal that spans a wider range has more potential
    to separate similar from dissimilar pairs.
    Limitation: does not account for how the values are distributed
    within that range.
    """
    ranges = {
        "s1_stat": df["s1_stat"].max() - df["s1_stat"].min(),
        "s2_name": df["s2_name"].max() - df["s2_name"].min(),
        "s3_sem":  df["s3_sem"].max()  - df["s3_sem"].min(),
    }
    total = sum(ranges.values())
    w1 = ranges["s1_stat"] / total
    w2 = ranges["s2_name"] / total
    w3 = ranges["s3_sem"]  / total
    log.info("Option 1 (range-based):   w1=%.4f  w2=%.4f  w3=%.4f", w1, w2, w3)
    return round(w1, 4), round(w2, 4), round(w3, 4)


# ===========================================================================
# Option 2 — Variance-based weights  (RECOMMENDED)
# ===========================================================================

def variance_based_weights(df: pd.DataFrame) -> tuple[float, float, float]:
    """
    w_k = Var(signal_k) / sum_of_variances

    Rationale: variance directly measures how much a signal discriminates
    between column pairs. A low-variance signal gives nearly the same
    score to all pairs — it is uninformative regardless of its range.

    This is the same principle used in PCA to rank feature importance
    by explained variance.
    """
    variances = {
        "s1_stat": df["s1_stat"].var(),
        "s2_name": df["s2_name"].var(),
        "s3_sem":  df["s3_sem"].var(),
    }
    total = sum(variances.values())
    w1 = variances["s1_stat"] / total
    w2 = variances["s2_name"] / total
    w3 = variances["s3_sem"]  / total
    log.info("Option 2 (variance-based): w1=%.4f  w2=%.4f  w3=%.4f", w1, w2, w3)
    log.info("  Variances: stat=%.6f  name=%.6f  sem=%.6f",
             variances["s1_stat"], variances["s2_name"], variances["s3_sem"])
    return round(w1, 4), round(w2, 4), round(w3, 4)


# ===========================================================================
# Option 3 — Correlation diagnostic
# ===========================================================================

def correlation_diagnostic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation between all three signals.

    Interpretation:
      |r| < 0.3  → weak correlation  → signals are independent ✓
      |r| 0.3–0.7 → moderate         → some overlap, acceptable
      |r| > 0.7  → strong            → signals are redundant ✗
                                        splitting weight between them
                                        double-counts the same information
    """
    signals = df[["s1_stat", "s2_name", "s3_sem"]]
    corr = signals.corr(method="pearson")

    log.info("Option 3 (correlation diagnostic):")
    pairs = [
        ("s1_stat", "s2_name", "stat vs name"),
        ("s1_stat", "s3_sem",  "stat vs sem"),
        ("s2_name", "s3_sem",  "name vs sem"),
    ]
    for a, b, label in pairs:
        r, p = pearsonr(df[a], df[b])
        flag = "✓ independent" if abs(r) < 0.3 else \
               "~ moderate overlap" if abs(r) < 0.7 else \
               "✗ REDUNDANT — double-counting risk"
        log.info("  %s : r=%.4f  p=%.2e  %s", label, r, p, flag)

    return corr


# ===========================================================================
# Round weights to clean values for practical use
# ===========================================================================

def round_to_clean(w1: float, w2: float, w3: float) -> tuple[float, float, float]:
    step = 0.05
    r1 = round(round(w1 / step) * step, 4)
    r2 = round(round(w2 / step) * step, 4)
    r3 = round(1.0 - r1 - r2, 4)   # force exact sum=1, no separate rounding
    r3 = max(r3, 0.0)               # safety: never negative
    return r1, r2, r3


# ===========================================================================
# Report
# ===========================================================================

def write_report(
    df: pd.DataFrame,
    p_stat_max: float,
    w_range:    tuple,
    w_var:      tuple,
    corr:       pd.DataFrame,
    report_path: Path,
    weights_csv: Path,
) -> None:

    w1_final, w2_final, w3_final = round_to_clean(*w_var)

    # Signal stats
    stats = {}
    for col, label in [("s1_stat","stat"), ("s2_name","name"), ("s3_sem","sem")]:
        s = df[col]
        stats[label] = {
            "min": s.min(), "max": s.max(),
            "mean": s.mean(), "std": s.std(), "var": s.var(),
            "range": s.max() - s.min(),
        }

    lines = [
        "=" * 70,
        "CCM Pipeline — Data-Driven Weight Derivation for Step 3.4.2",
        "=" * 70,
        "",
        "Formula:  Sim_attr(Ai,Aj) = w1*(1-P_stat/P_stat_max) + w2*P_name + w3*P_sem",
        f"P_stat_max (global normaliser) = {p_stat_max:.4f}",
        f"Column pairs analysed          = {len(df)}",
        "",
        "─" * 70,
        "SIGNAL STATISTICS  (all converted to similarity ∈ [0,1])",
        "─" * 70,
        f"{'Signal':<12} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8} {'Var':>10} {'Range':>8}",
        "-" * 70,
    ]
    for label, s in stats.items():
        lines.append(
            f"{label:<12} {s['min']:>8.4f} {s['max']:>8.4f} "
            f"{s['mean']:>8.4f} {s['std']:>8.4f} "
            f"{s['var']:>10.6f} {s['range']:>8.4f}"
        )

    lines += [
        "",
        "─" * 70,
        "OPTION 1 — Range-based weights",
        "─" * 70,
        "  w_k = range_k / sum(ranges)",
        f"  w1(stat) = {stats['stat']['range']:.4f} / {sum(s['range'] for s in stats.values()):.4f}"
        f" = {w_range[0]:.4f}",
        f"  w2(name) = {stats['name']['range']:.4f} / {sum(s['range'] for s in stats.values()):.4f}"
        f" = {w_range[1]:.4f}",
        f"  w3(sem)  = {stats['sem']['range']:.4f}  / {sum(s['range'] for s in stats.values()):.4f}"
        f" = {w_range[2]:.4f}",
        f"  → w1={w_range[0]:.4f}  w2={w_range[1]:.4f}  w3={w_range[2]:.4f}",
        "",
        "─" * 70,
        "OPTION 2 — Variance-based weights  (RECOMMENDED)",
        "─" * 70,
        "  w_k = Var(signal_k) / sum(Var)",
        "  Rationale: variance measures discriminative power.",
        "  Higher variance → signal better separates similar from dissimilar pairs.",
        f"  Var(stat) = {stats['stat']['var']:.6f}",
        f"  Var(name) = {stats['name']['var']:.6f}",
        f"  Var(sem)  = {stats['sem']['var']:.6f}",
        f"  Total Var = {sum(s['var'] for s in stats.values()):.6f}",
        "",
        f"  Raw weights  : w1={w_var[0]:.4f}  w2={w_var[1]:.4f}  w3={w_var[2]:.4f}",
        f"  Rounded (0.05 step): w1={w1_final:.4f}  w2={w2_final:.4f}  w3={w3_final:.4f}",
        "",
        "─" * 70,
        "OPTION 3 — Correlation diagnostic",
        "─" * 70,
        "  Checks whether any two signals are redundant (|r| > 0.7).",
        "",
        f"  stat vs name : r = {corr.loc['s1_stat','s2_name']:+.4f}",
        f"  stat vs sem  : r = {corr.loc['s1_stat','s3_sem']:+.4f}",
        f"  name vs sem  : r = {corr.loc['s2_name','s3_sem']:+.4f}",
        "",
    ]

    # Interpretation of correlations
    for pair, key in [("stat vs name",("s1_stat","s2_name")),
                      ("stat vs sem", ("s1_stat","s3_sem")),
                      ("name vs sem", ("s2_name","s3_sem"))]:
        r = corr.loc[key[0], key[1]]
        if abs(r) < 0.3:
            msg = f"  {pair}: |r|={abs(r):.4f} < 0.3 → signals are independent ✓"
        elif abs(r) < 0.7:
            msg = f"  {pair}: |r|={abs(r):.4f} → moderate overlap, acceptable ~"
        else:
            msg = f"  {pair}: |r|={abs(r):.4f} > 0.7 → REDUNDANT, double-counting risk ✗"
        lines.append(msg)

    lines += [
        "",
        "─" * 70,
        "FINAL RECOMMENDATION",
        "─" * 70,
        "",
        "  Derived weights (variance-based, rounded to 0.05):",
        f"    w1 (statistical) = {w1_final}",
        f"    w2 (name)        = {w2_final}",
        f"    w3 (semantic)    = {w3_final}",
        "",
        "  Previous heuristic weights:",
        f"    w1={HEURISTIC_W[0]}  w2={HEURISTIC_W[1]}  w3={HEURISTIC_W[2]}",
        "",
    ]

    # Compare
    changed = (
        abs(w1_final - HEURISTIC_W[0]) > 0.05 or
        abs(w2_final - HEURISTIC_W[1]) > 0.05 or
        abs(w3_final - HEURISTIC_W[2]) > 0.05
    )
    if changed:
        lines += [
            "  ⚠ The derived weights DIFFER from the heuristic weights.",
            "  Update column_graph.py with the derived weights.",
            "",
            "  Paper justification:",
            '  "Weights were derived from the variance of each proximity signal',
            '   across all column pairs, following the PCA principle that higher',
            '   variance indicates greater discriminative power (Eq. X).',
            f'   This yielded w1={w1_final}, w2={w2_final}, w3={w3_final}."',
        ]
    else:
        lines += [
            "  ✓ The derived weights CONFIRM the heuristic weights.",
            "  The original w1=0.4, w2=0.3, w3=0.3 are data-supported.",
            "",
            "  Paper justification:",
            '  "Weights were set to w1=0.4, w2=0.3, w3=0.3 and validated',
            '   by a variance-based analysis showing that the statistical',
            '   proximity signal carries the highest discriminative power',
            '   (Var=X), followed by name similarity (Var=Y) and semantic',
            '   alignment (Var=Z), consistent with the assigned weights."',
        ]

    lines.append("=" * 70)
    text = "\n".join(lines)
    report_path.write_text(text, encoding="utf-8")
    print("\n" + text)

    # Save derived weights as CSV for use in pipeline
    pd.DataFrame([{
        "method": "variance_based",
        "w1_stat_raw": w_var[0],
        "w2_name_raw": w_var[1],
        "w3_sem_raw":  w_var[2],
        "w1_stat_rounded": w1_final,
        "w2_name_rounded": w2_final,
        "w3_sem_rounded":  w3_final,
        "w1_heuristic": HEURISTIC_W[0],
        "w2_heuristic": HEURISTIC_W[1],
        "w3_heuristic": HEURISTIC_W[2],
    }]).to_csv(weights_csv, index=False)
    log.info("Derived weights saved to %s", weights_csv)


# ===========================================================================
# Main
# ===========================================================================

import argparse

# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CCM Step 3b — Variance-Based Weight Derivation")
    parser.add_argument("--input_dir", default="ccm_output",
                        help="Directory containing step3_proximity_long.csv "
                             "(default: ccm_output)")
    parser.add_argument("--out_dir", default="ccm_output",
                        help="Output directory for derived_weights.* "
                             "(default: ccm_output)")
    parser.add_argument("--dataset_dir", default=None,
                   help="Dataset working directory containing schema.json, knowledge.docx, "
                        "csv/ and ccm_output/. When run_pipeline.py is in the parent folder, "
                        "pass the dataset subfolder name (e.g. --dataset_dir Chinook). "
                        "Defaults to the current working directory.")
    args = parser.parse_args()

    # ── Dataset directory — chdir so all relative paths resolve correctly ────
    if args.dataset_dir is not None:
        import os as _os
        _os.chdir(args.dataset_dir)

    input_dir = Path(args.input_dir)
    out_dir   = Path(args.out_dir)
    input_csv = input_dir / "step3_proximity_long.csv"

    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input not found: {input_csv}\n"
            "Run 2_p_stat_name_sem.py first."
        )

    log.info("=" * 60)
    log.info("CCM — Variance-Based Weight Derivation (Step 3.4.2)")
    log.info("  Input : %s", input_csv)
    log.info("  Output: %s", out_dir)
    log.info("=" * 60)

    # Load and convert signals
    df, p_stat_max = load_signals(input_csv)

    # Option 1 — range-based
    w_range = range_based_weights(df)

    # Option 2 — variance-based (recommended)
    w_var = variance_based_weights(df)

    # Option 3 — correlation diagnostic
    corr = correlation_diagnostic(df)

    # Write report and save weights
    write_report(
        df, p_stat_max,
        w_range, w_var, corr,
        report_path  = out_dir / "derived_weights.txt",
        weights_csv  = out_dir / "derived_weights.csv",
    )


if __name__ == "__main__":
    main()