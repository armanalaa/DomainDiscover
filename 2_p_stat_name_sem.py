"""
=============================================================================
CCM Pipeline — Step 3  (P_stat · P_name · P_sem)
=============================================================================

Reads outputs of Steps 1 & 2 and computes three pairwise proximity scores
for every column pair (Ai, Aj):

  Step 3.1  P_stat(Ai,Aj)  —  Statistical proximity           (Alserafi Eq. 3)
  Step 3.2  P_name(Ai,Aj)  —  Name similarity                 (Alserafi Eq. 4)
  Step 3.3  P_sem(Ai,Aj)   —  Concept-based semantic alignment (Running Example)

─────────────────────────────────────────────────────────────────────────────
FORMULAS
─────────────────────────────────────────────────────────────────────────────

  3.1  P_stat(Ai, Aj) = (1/k) * Σ_l | z_l(Ai) − z_l(Aj) |
         where z_l is the z-scored value of feature l (from step2_column_profiles.json)
         k = number of features (17, Alserafi Table 3)
         → small value means statistically similar

  3.2  P_name(Ai, Aj) = 1 − Lev(Ai.name, Aj.name) / max(|Ai.name|, |Aj.name|)
         Levenshtein similarity normalised to [0, 1]
         → large value means names are lexically similar

  3.3  P_sem(Ai, Aj)  = max_{c ∈ C} [ φ_i[c] × φ_j[c] ]
         φ_i[c] = cosine(e(Ai), e(c))   already in phi_matrix.csv
         → large value means both columns strongly align with the SAME concept

─────────────────────────────────────────────────────────────────────────────
NOTE ON P_stat DIRECTION
─────────────────────────────────────────────────────────────────────────────
  P_stat is a DISTANCE (lower = more similar), while P_name and P_sem are
  SIMILARITIES (higher = more similar).  All three are kept in their natural
  form here so that Step 3.4 (Random Forest) can learn the correct direction
  from labeled examples.  The feature vector fed to RF is:

    x_ij = [ P_stat, P_name, P_sem ]

  The RF learns:
    if P_stat LOW  and P_sem HIGH  →  likely related
    if P_stat HIGH and P_sem LOW   →  likely unrelated

─────────────────────────────────────────────────────────────────────────────
INSTALL  (run once)
─────────────────────────────────────────────────────────────────────────────
  pip install numpy pandas

  NOTE: no sentence-transformers needed here — embeddings were already
  computed in Step 2 and are stored in step2_column_profiles.json and
  phi_matrix.csv.  This step is pure numpy/pandas.

─────────────────────────────────────────────────────────────────────────────
INPUTS  (from ccm_output/)
─────────────────────────────────────────────────────────────────────────────
  step2_column_profiles.json   → z_features per column  (for P_stat)
                               → column names           (for P_name)
                               → embeddings             (for P_sem fallback)
  phi_matrix.csv               → φ_i[c] matrix          (for P_sem, preferred)

─────────────────────────────────────────────────────────────────────────────
OUTPUTS  (in ccm_output/)
─────────────────────────────────────────────────────────────────────────────
  step3_P_stat.csv     — (N_cols × N_cols) pairwise statistical distance
  step3_P_name.csv     — (N_cols × N_cols) pairwise name similarity
  step3_P_sem.csv      — (N_cols × N_cols) pairwise semantic alignment
  step3_proximity_long.csv — all three scores in long format (one row per pair)
                             columns: col_i, col_j, P_stat, P_name, P_sem
                             → direct input to Step 3.4 (Random Forest)

─────────────────────────────────────────────────────────────────────────────
RUN
─────────────────────────────────────────────────────────────────────────────
  python ccm_step3_proximity.py
  python ccm_step3_proximity.py --input-dir ./ccm_output --out-dir ./ccm_output
=============================================================================
"""

from __future__ import annotations
import os

import argparse
import json
import logging
import math
import time
from itertools import combinations
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    INPUT_DIR: Path = Path("ccm_output")
    OUT_DIR:   Path = Path("ccm_output")

    # Files produced by Steps 1 & 2
    PROFILES_FILE: str = "step2_column_profiles.json"
    PHI_FILE:      str = "phi_matrix.csv"

    # Output files
    P_STAT_FILE:  str = "step3_P_stat.csv"
    P_NAME_FILE:  str = "step3_P_name.csv"
    P_SEM_FILE:   str = "step3_P_sem.csv"
    LONG_FILE:    str = "step3_proximity_long.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Z-FEATURE KEYS  (same order as step2, Alserafi Table 3)
# ─────────────────────────────────────────────────────────────────────────────
Z_FEATURE_KEYS = [
    # universal (3)
    "distinct_values_cnt", "distinct_values_pct", "missing_values_pct",
    # nominal (8)
    "val_size_avg", "val_size_min", "val_size_max", "val_size_std",
    "val_pct_median", "val_pct_min", "val_pct_max", "val_pct_std",
    # numeric (6)
    "mean", "std", "min_val", "max_val", "range_val", "co_of_var",
]


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_profiles(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        profiles = json.load(f)
    log.info("Loaded %d column profiles from %s", len(profiles), path.name)
    return profiles


def load_phi_matrix(path: Path) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Load phi_matrix.csv produced by Step 2.

    Returns:
      Phi      — (N_cols, N_concepts) numpy array  φ_i[c]
      col_ids  — list of "Table.column" row labels
      con_ids  — list of concept IDs  (C1 … C7)
    """
    if not path.exists():
        log.warning("phi_matrix.csv not found — P_sem will be computed from "
                    "raw embeddings in step2_column_profiles.json")
        return None, None, None

    if _PANDAS:
        df      = pd.read_csv(path, index_col=0)
        col_ids = list(df.index)
        con_ids = list(df.columns)
        Phi     = df.values.astype(np.float64)
    else:
        with open(path, encoding="utf-8") as f:
            header  = f.readline().strip().split(",")
            con_ids = header[1:]
            col_ids = []
            rows    = []
            for line in f:
                parts = line.strip().split(",")
                col_ids.append(parts[0])
                rows.append([float(x) for x in parts[1:]])
        Phi = np.array(rows, dtype=np.float64)

    log.info("Loaded phi matrix: %d columns × %d concepts from %s",
             *Phi.shape, path.name)
    return Phi, col_ids, con_ids


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3.1  —  P_stat  (Alserafi Eq. 3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_P_stat(profiles: list[dict]) -> np.ndarray:
    """
    Step 3.1 — Statistical Proximity

    Formula (Running Example + Alserafi Eq. 3):

      P_stat(Ai, Aj) = (1/k) * Σ_l | z_l(Ai) − z_l(Aj) |

    where:
      z_l(Ai)  = z-scored value of feature l for column Ai
                 (pre-computed globally in Step 2, stored in z_features)
      k        = number of features = 17  (Alserafi Table 3)

    Interpretation:
      → P_stat is a DISTANCE:  small value = statistically similar columns
      → Range: [0, ∞)  but in practice most pairs fall in [0, 3]

    Running Example:
      Profile(A6) = [1.0, -0.2,  0.8, -0.3]   (z_distinct, z_missing, ...)
      Profile(A7) = [0.5, -0.1,  0.7, -0.4]
      |diff|      = [0.5,  0.1,  0.1,  0.1]
      P_stat      = (0.5 + 0.1 + 0.1 + 0.1) / 4 = 0.20  → similar

    Returns:
      P_stat_matrix — (N, N) numpy array, symmetric, diagonal = 0
    """
    log.info("Step 3.1 — Computing P_stat  (mean |z_i − z_j| over %d features)",
             len(Z_FEATURE_KEYS))

    N = len(profiles)
    k = len(Z_FEATURE_KEYS)

    # Build Z matrix: shape (N, k)
    # z_features[feat] is 0.0 for inapplicable features (e.g. mean for nominal)
    Z = np.zeros((N, k), dtype=np.float64)
    for i, p in enumerate(profiles):
        zf = p.get("z_features", {})
        for j, feat in enumerate(Z_FEATURE_KEYS):
            Z[i, j] = zf.get(feat, 0.0)

    # P_stat(i, j) = mean of |Z[i] − Z[j]|  for all feature positions
    # Vectorised: compute pairwise L1 distance / k
    # For 130 columns this is a 130×130 matrix — trivially fast
    P_stat = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        diff = np.abs(Z[i] - Z)          # (N, k)
        P_stat[i] = diff.mean(axis=1)    # (N,)

    log.info("  P_stat computed — shape %s, range [%.4f, %.4f]",
             P_stat.shape, P_stat.min(), P_stat.max())

    # Spot-check: print 3 examples
    _log_examples("P_stat (distance — lower = more similar)",
                  P_stat, profiles, higher_is_similar=False, n=3)
    return P_stat


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3.2  —  P_name  (Alserafi Eq. 4)
# ─────────────────────────────────────────────────────────────────────────────

def _levenshtein(s1: str, s2: str) -> int:
    """
    Standard dynamic-programming Levenshtein edit distance.
    No external library needed.
    """
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s2) + 1))
    for ch1 in s1:
        curr = [prev[0] + 1]
        for j, ch2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1,   # deletion
                            curr[j]     + 1,   # insertion
                            prev[j]     + (ch1 != ch2)))  # substitution
        prev = curr
    return prev[-1]


def compute_P_name(profiles: list[dict]) -> np.ndarray:
    """
    Step 3.2 — Column Name Similarity

    Formula (Running Example + Alserafi Eq. 4):

      P_name(Ai, Aj) = 1 − Lev(Ai.name, Aj.name) / max(|Ai.name|, |Aj.name|)

    where:
      Lev(·,·)   = Levenshtein edit distance (pure Python — no library needed)
      |name|     = length of name string in characters

    Interpretation:
      → P_name is a SIMILARITY:  large value = lexically similar names
      → Range: [0, 1]
      → P_name = 1.0  iff names are identical

    Running Example:
      order_id        (len=8)
      order_date      (len=10)
      Lev = 5
      P_name = 1 − 5/10 = 1 − 0.50 = 0.50  (moderate similarity)

    Note: column names are compared as-is (no lowercasing) to stay faithful
    to the running example.  You may lowercase before calling if preferred.

    Returns:
      P_name_matrix — (N, N) numpy array, symmetric, diagonal = 1
    """
    log.info("Step 3.2 — Computing P_name  (normalised Levenshtein similarity)")

    N     = len(profiles)
    names = [p["column"] for p in profiles]

    P_name = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        P_name[i, i] = 1.0
        for j in range(i + 1, N):
            lev      = _levenshtein(names[i], names[j])
            max_len  = max(len(names[i]), len(names[j]))
            sim      = 1.0 - lev / max_len if max_len > 0 else 1.0
            P_name[i, j] = sim
            P_name[j, i] = sim

    log.info("  P_name computed — shape %s, range [%.4f, %.4f]",
             P_name.shape, P_name.min(), P_name.max())

    _log_examples("P_name (similarity — higher = more similar)",
                  P_name, profiles, higher_is_similar=True, n=3)
    return P_name


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3.3  —  P_sem  (Running Example §3.3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_P_sem(profiles: list[dict],
                  Phi:      np.ndarray | None,
                  col_ids:  list[str]  | None) -> np.ndarray:
    """
    Step 3.3 — Concept-Based Semantic Alignment Score

    Formula (Running Example §3.3.1–3.3.3):

      Step 3.3.1  φ_i[c] = cosine(e(Ai), e(c))   ← already in phi_matrix.csv
      Step 3.3.2  score[c] = φ_i[c] × φ_j[c]     (element-wise product)
      Step 3.3.3  P_sem(Ai, Aj) = max_{c ∈ C} score[c]

    Rationale (Running Example):
      Multiplication ensures that if one column aligns strongly to a concept
      but the other does not, the score stays low:
        e.g.  0.9 × 0.1 = 0.09  (not similar)
      Both columns must align to the SAME concept to get a high score:
        e.g.  1.0 × 0.98 = 0.98  (semantically similar)

    Running Example:
      φ_i[C1] = 1.0,  φ_j[C1] = 0.98  →  score[C1] = 0.98
      φ_i[C2] = 0.30, φ_j[C2] = 0.25  →  score[C2] = 0.075
      P_sem = max(0.98, 0.075) = 0.98   (both belong to the same concept)

    Two sources for φ_i[c]:
      PRIMARY   — phi_matrix.csv  (computed from real embeddings in Step 2)
      FALLBACK  — raw embeddings in step2_column_profiles.json  (recomputed)

    Returns:
      P_sem_matrix — (N, N) numpy array, symmetric, diagonal = 1
    """
    log.info("Step 3.3 — Computing P_sem  (concept-based semantic alignment)")

    N = len(profiles)

    # ── Prefer phi_matrix.csv (avoids re-loading embeddings) ────────────────
    if Phi is not None and col_ids is not None:
        # Build lookup: "Table.column" → row index in Phi
        phi_index = {label: idx for idx, label in enumerate(col_ids)}

        # Map each profile to its row in Phi
        # profile label = "Table.column"
        Phi_ordered = np.zeros((N, Phi.shape[1]), dtype=np.float64)
        missing     = []
        for i, p in enumerate(profiles):
            label = f"{p['table']}.{p['column']}"
            idx   = phi_index.get(label)
            if idx is not None:
                Phi_ordered[i] = Phi[idx]
            else:
                missing.append(label)

        if missing:
            log.warning("  %d columns not found in phi_matrix — "
                        "their φ rows will be zero: %s",
                        len(missing), missing[:5])

        log.info("  Using phi_matrix.csv  (%d cols × %d concepts)",
                 *Phi_ordered.shape)

    # ── Fallback: compute φ from raw embeddings stored in profiles ───────────
    else:
        log.info("  phi_matrix.csv unavailable — "
                 "computing phi from raw embeddings in profiles")

        # Collect column embeddings
        embs = [p.get("embedding", []) for p in profiles]
        if not any(embs):
            log.error("  No embeddings found in profiles either. "
                      "Run Step 2 with sentence-transformers installed.")
            return np.zeros((N, N), dtype=np.float64)

        E_col = np.array(embs, dtype=np.float64)   # (N, 768)

        # We would need concept embeddings too — not available here
        # without step1_concepts.json having real embeddings
        log.error("  Concept embeddings required for fallback P_sem. "
                  "Please run Step 2 with sentence-transformers to generate "
                  "phi_matrix.csv, then re-run Step 3.")
        return np.zeros((N, N), dtype=np.float64)

    # ── P_sem(i, j) = max_c [ φ_i[c] * φ_j[c] ] ────────────────────────────
    #
    # Vectorised form:
    #   For each pair (i,j) and concept c: score[c] = Phi[i,c] * Phi[j,c]
    #   P_sem[i,j] = max over c
    #
    # Matrix form (fast, no Python loop over pairs):
    #   Element-wise product:  Phi[i] * Phi[j]  for all j simultaneously
    #   = Phi_ordered[i, :] * Phi_ordered  →  (N, C)
    #   max over C axis = (N,)
    #   → build full (N, N) matrix row by row

    P_sem = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        # Phi_ordered[i]  shape (C,)
        # Phi_ordered      shape (N, C)
        # product          shape (N, C)  — each row j is φ_i[c]*φ_j[c] for all c
        product    = Phi_ordered[i] * Phi_ordered     # (N, C)
        P_sem[i]   = product.max(axis=1)              # (N,)

    # Ensure symmetry (numerical safety)
    P_sem = (P_sem + P_sem.T) / 2.0
    np.fill_diagonal(P_sem, 1.0)

    log.info("  P_sem computed — shape %s, range [%.4f, %.4f]",
             P_sem.shape, P_sem.min(), P_sem.max())

    _log_examples("P_sem (similarity — higher = more semantically aligned)",
                  P_sem, profiles, higher_is_similar=True, n=3)
    return P_sem


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _log_examples(title: str, M: np.ndarray,
                  profiles: list[dict],
                  higher_is_similar: bool,
                  n: int = 3) -> None:
    """Print the n most-similar and n most-different off-diagonal pairs."""
    N = len(profiles)
    # collect upper-triangle pairs
    pairs = [(i, j, M[i, j])
             for i in range(N) for j in range(i + 1, N)]

    pairs.sort(key=lambda x: x[2], reverse=higher_is_similar)

    log.info("  ── %s", title)
    log.info("  Top-%d most SIMILAR pairs:", n)
    for i, j, v in pairs[:n]:
        log.info("    %-35s  %-35s  %.4f",
                 f"{profiles[i]['table']}.{profiles[i]['column']}",
                 f"{profiles[j]['table']}.{profiles[j]['column']}", v)
    log.info("  Top-%d most DIFFERENT pairs:", n)
    for i, j, v in pairs[-n:]:
        log.info("    %-35s  %-35s  %.4f",
                 f"{profiles[i]['table']}.{profiles[i]['column']}",
                 f"{profiles[j]['table']}.{profiles[j]['column']}", v)


def _save_matrix(M: np.ndarray, profiles: list[dict], path: Path) -> None:
    """Save an (N×N) matrix as CSV with Table.column labels."""
    labels = [f"{p['table']}.{p['column']}" for p in profiles]
    if _PANDAS:
        pd.DataFrame(M, index=labels, columns=labels).to_csv(
            path, float_format="%.6f")
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write("," + ",".join(labels) + "\n")
            for lbl, row in zip(labels, M):
                f.write(lbl + "," +
                        ",".join(f"{v:.6f}" for v in row) + "\n")
    log.info("Saved %s  (%d × %d)", path.name, *M.shape)


def _save_long_format(P_stat: np.ndarray,
                      P_name: np.ndarray,
                      P_sem:  np.ndarray,
                      profiles: list[dict],
                      path: Path) -> None:
    """
    Save all three proximity scores in long format:
      col_i, col_j, table_i, table_j, domain_i, domain_j,
      P_stat, P_name, P_sem, same_domain

    Upper-triangle only (pairs are symmetric — no duplicates).
    This is the direct input to Step 3.4 (Random Forest).
    """
    N = len(profiles)
    rows = []
    for i in range(N):
        for j in range(i + 1, N):
            pi, pj = profiles[i], profiles[j]
            rows.append({
                "col_i":       f"{pi['table']}.{pi['column']}",
                "col_j":       f"{pj['table']}.{pj['column']}",
                "table_i":     pi["table"],
                "table_j":     pj["table"],
                "domain_i":    pi.get("domain", ""),
                "domain_j":    pj.get("domain", ""),
                "type_i":      "numeric" if pi["is_numeric"] else "nominal",
                "type_j":      "numeric" if pj["is_numeric"] else "nominal",
                "P_stat":      round(float(P_stat[i, j]), 6),
                "P_name":      round(float(P_name[i, j]), 6),
                "P_sem":       round(float(P_sem[i, j]),  6),
                # same_domain = ground-truth label proxy for Step 3.4
                # 1 if both columns belong to the same domain, 0 otherwise
                "same_domain": int(pi.get("domain","") == pj.get("domain","")
                                   and pi.get("domain","") != ""),
            })

    if _PANDAS:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        if not rows:
            return
        header = list(rows[0].keys())
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(str(r[h]) for h in header) + "\n")

    total  = len(rows)
    pos    = sum(r["same_domain"] for r in rows)
    log.info("Saved %s  (%d pairs, %d same-domain, %d cross-domain)",
             path.name, total, pos, total - pos)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CCM Step 3 — P_stat, P_name, P_sem")
    parser.add_argument("--input-dir",  default=str(Config.INPUT_DIR))
    parser.add_argument("--out-dir",    default=str(Config.OUT_DIR))
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

    Config.INPUT_DIR = Path(args.input_dir)
    Config.OUT_DIR   = Path(args.out_dir)
    Config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log.info("CCM Pipeline — Step 3: Column Proximity Computation")
    log.info("  Input dir : %s", Config.INPUT_DIR)
    log.info("  Output dir: %s", Config.OUT_DIR)

    # ── Load inputs ──────────────────────────────────────────────────────────
    profiles = load_profiles(Config.INPUT_DIR / Config.PROFILES_FILE)
    Phi, col_ids, con_ids = load_phi_matrix(Config.INPUT_DIR / Config.PHI_FILE)

    N = len(profiles)
    n_pairs = N * (N - 1) // 2
    log.info("Computing %d pairwise scores for %d columns (%d pairs)",
             3, N, n_pairs)

    # ── Step 3.1  P_stat ─────────────────────────────────────────────────────
    P_stat = compute_P_stat(profiles)
    _save_matrix(P_stat, profiles, Config.OUT_DIR / Config.P_STAT_FILE)

    # ── Step 3.2  P_name ─────────────────────────────────────────────────────
    P_name = compute_P_name(profiles)
    _save_matrix(P_name, profiles, Config.OUT_DIR / Config.P_NAME_FILE)

    # ── Step 3.3  P_sem ──────────────────────────────────────────────────────
    P_sem = compute_P_sem(profiles, Phi, col_ids)
    _save_matrix(P_sem, profiles, Config.OUT_DIR / Config.P_SEM_FILE)

    # ── Long format (Step 3.4 input) ─────────────────────────────────────────
    _save_long_format(P_stat, P_name, P_sem, profiles,
                      Config.OUT_DIR / Config.LONG_FILE)

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("━" * 66)
    log.info("STEP 3 COMPLETE  (%.1fs)", time.time() - t0)
    log.info("━" * 66)
    log.info("  Columns        : %d", N)
    log.info("  Pairs computed : %d", n_pairs)
    log.info("")
    log.info("  P_stat  range  : [%.4f, %.4f]  (distance,   lower = more similar)",
             float(P_stat[np.triu_indices(N,1)].min()),
             float(P_stat[np.triu_indices(N,1)].max()))
    log.info("  P_name  range  : [%.4f, %.4f]  (similarity, higher = more similar)",
             float(P_name[np.triu_indices(N,1)].min()),
             float(P_name[np.triu_indices(N,1)].max()))
    if P_sem.any():
        log.info("  P_sem   range  : [%.4f, %.4f]  (similarity, higher = more similar)",
                 float(P_sem[np.triu_indices(N,1)].min()),
                 float(P_sem[np.triu_indices(N,1)].max()))
    log.info("")
    log.info("Outputs:")
    log.info("  %-40s  P_stat  matrix  (%d×%d)",
             Config.P_STAT_FILE, N, N)
    log.info("  %-40s  P_name  matrix  (%d×%d)",
             Config.P_NAME_FILE, N, N)
    log.info("  %-40s  P_sem   matrix  (%d×%d)",
             Config.P_SEM_FILE, N, N)
    log.info("  %-40s  long-format input for Step 3.4 RF",
             Config.LONG_FILE)
    log.info("")
    log.info("Feature vector per pair  →  x_ij = [P_stat, P_name, P_sem]")
    log.info("Next: Step 3.4 — Random Forest → Sim_attr(Ai, Aj)")


if __name__ == "__main__":
    main()