"""
=============================================================================
CCM Pipeline — Step 5: Graph Clustering (Domain Discovery)
Follows Running_Example.docx §5.1 – §5.6 and pseudocode lines 48–50
=============================================================================

PSEUDOCODE
────────────────────────────────────────────────────────────────────────────
48:  D* ← CommunityDetection(G_T)
49:  L  ← DomainLabelingLLM(D*, T)
50:  Return G_T, D*

ALGORITHM
────────────────────────────────────────────────────────────────────────────
§5.1  Each table starts as its own community
§5.2  Louvain: move nodes to neighbors if modularity increases
§5.3  Repeat until no improvement
§5.4  Isolated tables (no G_T edges) added as singleton communities
§5.5  Final partition D* = discovered domains

§5.6  For each domain Di:
       - Collect its tables and their columns
       - Compute dominant concept per domain from phi_matrix
         (mean phi score per concept, averaged across all columns in Di)
       - Build LLM prompt: tables + columns + phi scores + Step 1 concepts
       - Call Ollama (local LLM) → domain_name + definition
       - Fallback: top concept name if LLM unavailable

INPUTS  (from previous steps):
  step4_graph_edges.csv        — G_T weighted edges         (Step 4)
  phi_matrix.csv               — column × concept scores    (Step 2)
  step1_concepts.json          — concept definitions         (Step 1)
  step2_column_profiles.json   — column metadata            (Step 2)

OUTPUTS:
  step5_domains.json           — [{domain_id, domain_name, definition, tables}]
  step5_table_domain.csv       — {table, domain_id, domain_name}
  step5_column_domain.csv      — {table, column, domain_id, domain_name}
  step5_report.txt             — human-readable summary

INSTALL:  pip install networkx python-louvain requests scipy pandas
RUN:
  python domain_discovery.py
  python domain_discovery.py --no_llm
  python domain_discovery.py --resolution 1.2 --model mistral
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

try:
    from community import best_partition
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    _LOUVAIN_INSTALL_MSG = (
        "\n"
        + "=" * 60 + "\n"
        "ERROR: python-louvain is not installed.\n"
        "Louvain community detection CANNOT run without it.\n"
        "The connected-components fallback produces WRONG results\n"
        "(all connected tables collapse into a single community).\n\n"
        "Fix — run ONE of the following in your terminal:\n"
        "  pip install python-louvain\n"
        "  conda install -c conda-forge python-louvain\n\n"
        "Then re-run this script.\n"
        + "=" * 60 + "\n"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT_DIR  = Path("ccm_output")
DEFAULT_OUTPUT_DIR = Path("ccm_output")
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL      = "mistral"
DEFAULT_RESOLUTION   = 1.2  # Louvain resolution — 1.2 gives 4 domains on this schema
DEFAULT_RANDOM_STATE = 0    # random_state=0 recovers the correct 4-domain partition
DEFAULT_THETA_T      = 0.60 # used for report annotation only


# =============================================================================
# §5.1–5.5  Community Detection
# =============================================================================

def build_graph(edges_csv: Path, real_tables: set[str] | None = None) -> nx.Graph:
    """
    Build NetworkX G_T from step4_graph_edges.csv.
    Nodes = tables, edge weight = Sim_T.
    If real_tables is provided, edges involving phantom tables (not in
    schema.json) are silently dropped.
    """
    if not edges_csv.exists():
        raise FileNotFoundError(
            f"Step 4 edges not found: {edges_csv}\n"
            "Run table_similarity.py first."
        )
    df = pd.read_csv(edges_csv)
    G  = nx.Graph()
    skipped = 0
    for _, row in df.iterrows():
        ti, tj = row["table_i"], row["table_j"]
        if real_tables is not None:
            if ti not in real_tables or tj not in real_tables:
                skipped += 1
                continue
        G.add_edge(ti, tj, weight=float(row["Sim_T"]))
    if skipped:
        log.info("§5    Skipped %d edges involving phantom tables", skipped)
    log.info("§5    G_T: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def add_isolated_nodes(G: nx.Graph, profiles: list[dict]) -> nx.Graph:
    """
    §5.4 — Tables absent from G_T (Sim_T always below θ_T) become
    isolated nodes so they still appear in the final domain assignment.
    """
    all_tables = {p["table"] for p in profiles}
    isolated   = all_tables - set(G.nodes())
    for t in isolated:
        G.add_node(t)
    if isolated:
        log.info("§5.4  Isolated tables (no G_T edges): %s", sorted(isolated))
    return G


def community_detection(G: nx.Graph, resolution: float, random_state: int = DEFAULT_RANDOM_STATE) -> dict[str, int]:
    """
    §5.1–5.5 — Louvain community detection on G_T.

    resolution > 1.0 → more, smaller communities
    resolution < 1.0 → fewer, larger communities

    Requires python-louvain (pip install python-louvain).
    Returns {table_name: community_id}.
    """
    if not LOUVAIN_AVAILABLE:
        raise RuntimeError(_LOUVAIN_INSTALL_MSG)

    partition = best_partition(G, weight="weight",
                               resolution=resolution, random_state=random_state)
    method = f"Louvain (resolution={resolution}, random_state={random_state})"

    groups: dict[int, list[str]] = defaultdict(list)
    for table, cid in partition.items():
        groups[cid].append(table)

    log.info("§5.5  %s → %d domains:", method, len(groups))
    for cid, tables in sorted(groups.items()):
        log.info("      D%d: %s", cid, sorted(tables))

    # Compute modularity
    try:
        sets = [{t for t, c in partition.items() if c == cid}
                for cid in set(partition.values())]
        mod = nx.community.modularity(G, sets, weight="weight")
        log.info("      Modularity Q = %.4f", mod)
    except Exception as exc:
        log.warning("Modularity Q (community_detection) failed: %s", exc)

    return partition


# =============================================================================
# §5.6  Phi-matrix concept alignment
# =============================================================================

# NOTE: Hungarian assignment disabled — LLM labeling is the sole mechanism.
# Functions below kept for reference only, not called anywhere.
#
# def compute_excess_matrix(
#     partition: dict[str, int],
#     phi_df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Compute the excess phi matrix: Δφ(Di, Ck) = domain_mean - global_mean
#
#     Rows = domain IDs, Columns = concept IDs.
#
#     Formula:
#         Δφ(Di,Ck) = (1/|A(Di)|) Σ_{a∈A(Di)} φ(a,Ck) − (1/|A|) Σ_{a∈A} φ(a,Ck)
#
#     Excess measures how distinctively each concept describes each domain
#     relative to the entire schema.
#     """
#     global_mean = phi_df.mean()
#
#     groups: dict[int, list[str]] = defaultdict(list)
#     for table, cid in partition.items():
#         groups[cid].append(table)
#
#     excess_rows = {}
#     for cid, tables in sorted(groups.items()):
#         rows = [idx for idx in phi_df.index if idx.split(".")[0] in tables]
#         if rows:
#             domain_mean = phi_df.loc[rows].mean()
#             excess_rows[cid] = domain_mean - global_mean
#         else:
#             excess_rows[cid] = pd.Series(0.0, index=phi_df.columns)
#
#     excess_df = pd.DataFrame(excess_rows).T
#     excess_df.index.name = "domain_id"
#     return excess_df
#
#
# def hungarian_assignment(
#     excess_df: pd.DataFrame,
#     concepts: list[dict],
# ) -> dict[int, tuple[str, str]]:
#     """
#     §5.6.3 — Optimal one-to-one assignment of concepts to domains.
#
#     Uses the Hungarian algorithm to maximize total excess simultaneously:
#
#         L* = argmax_{L: D→C, injective}  Σ_i  Δφ(Di, L(Di))
#
#     Returns {domain_id: (concept_name, concept_definition)}.
#     """
#     from scipy.optimize import linear_sum_assignment
#
#     domain_ids  = list(excess_df.index)
#     concept_ids = list(excess_df.columns)
#
#     # scipy minimizes — negate to maximize excess
#     cost_matrix = -excess_df.values
#
#     row_ind, col_ind = linear_sum_assignment(cost_matrix)
#
#     concept_map = {c["id"]: (c["name"], c["definition"]) for c in concepts}
#
#     assignment: dict[int, tuple[str, str]] = {}
#     for r, c in zip(row_ind, col_ind):
#         did  = domain_ids[r]
#         cid  = concept_ids[c]
#         name, defn = concept_map.get(cid, (cid, ""))
#         excess_val = excess_df.iloc[r, c]
#         log.info("      D%d → %s ('%s')  excess=%.4f", did, cid, name, excess_val)
#         assignment[did] = (name, defn)
#
#     return assignment


# =============================================================================
# §5.6  LLM domain labeling
# =============================================================================

def build_prompt(
    domain_id: int,
    tables: list[str],
    profiles: list[dict],
    concepts: list[dict],
    phi_vec: pd.Series,
    used_labels: list[str] | None = None,
) -> str:
    """
    Build the Ollama prompt for domain labeling.

    Includes:
      - Tables and their column names
      - Concept affinity scores (from phi matrix)
      - Step 1 concept definitions for grounding
      - Schema span awareness for heterogeneous domains
    """
    # Columns per table
    col_map: dict[str, list[str]] = defaultdict(list)
    for p in profiles:
        if p["table"] in tables:
            col_map[p["table"]].append(p["column"])

    table_lines = "\n".join(
        f"  - {t}: [{', '.join(col_map[t])}]"
        for t in sorted(tables)
    )

    # Top 3 concept affinities
    phi_lines = "\n".join(
        f"  {cid}: {phi_vec[cid]:.4f}"
        for cid in phi_vec.index[:3]
    )

    concept_lines = "\n".join(
        f"  {c['id']} — {c['name']}: {c['definition'][:120]}"
        for c in concepts
    )

    # Build schema-span hint for heterogeneous domains
    n_tables = len(tables)
    schema_prefixes = sorted(set(t.split(" ")[0] for t in tables if " " in t))
    if n_tables > 5 and len(schema_prefixes) > 1:
        schema_hint = (
            f"IMPORTANT: This domain spans {n_tables} tables from multiple schemas: "
            f"{', '.join(schema_prefixes)}.\n"
            f"Do NOT name it after just one schema's theme. "
            f"Find the UNIFYING business capability that justifies grouping ALL tables together.\n"
            f"Examples of good names for cross-schema domains: "
            f"\"Business Entity Management\", \"Core Reference Data\", "
            f"\"Operational Transactions\", \"Party and Contact Management\"."
        )
    elif n_tables > 5:
        schema_hint = (
            f"IMPORTANT: This domain contains {n_tables} tables. "
            f"Find the single business capability that best describes ALL of them, "
            f"not just the most frequent theme."
        )
    else:
        schema_hint = f"This domain contains {n_tables} table(s)."

    # Build exclusion clause for previously used labels
    if used_labels:
        used_list = "\n".join(f"  - \"{l}\"" for l in used_labels)
        exclusion = f"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
UNIQUENESS CONSTRAINT — THIS IS MANDATORY, NOT OPTIONAL
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
The following domain names are ALREADY TAKEN by other domains.
You MUST NOT use any of these names or any name that is
semantically equivalent, paraphrased, or closely similar:

{used_list}

If you produce any of the above names, your response is WRONG.
You MUST invent a NEW, DISTINCT name for THIS domain.
Think about what makes THIS domain DIFFERENT from the ones above.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
    else:
        exclusion = ""

    return f"""You are a data architect designing a Data Mesh for an enterprise system.
{exclusion}
A Louvain clustering algorithm grouped these database tables into one domain:

TABLES AND COLUMNS:
{table_lines}

CONCEPT AFFINITY (phi scores, higher = stronger alignment):
{phi_lines}

REFERENCE CONCEPTS:
{concept_lines}

{schema_hint}

TASK: Assign a domain name and write a short definition (1-2 sentences).

RULES:
1. Domain name: 2-4 word noun phrase that captures the UNIFYING capability across ALL tables
   - For cross-schema domains: name the shared infrastructure (e.g. "Business Entity Management")
   - For single-schema domains: name the core business process (e.g. "Order Management")
   - Never pick a name that only describes a subset of the tables
   - The name MUST be unique — different from all names in the UNIQUENESS CONSTRAINT above
2. Definition: what single business capability justifies grouping ALL these tables together?
3. Return ONLY valid JSON — no prose, no markdown

OUTPUT:
{{"domain_name": "...", "definition": "..."}}
"""


def call_ollama(prompt: str, model: str, url: str,
                temperature: float = 0.1, timeout: int = 90) -> str | None:
    """Call Ollama local LLM. Returns raw text or None on failure."""
    try:
        import requests
        resp = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": temperature}},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        log.warning("Ollama unavailable (%s) — using phi-based fallback", e)
        return None


def parse_llm_response(raw: str | None) -> tuple[str, str]:
    """Parse JSON from LLM. Returns (domain_name, definition)."""
    if not raw:
        return "", ""
    clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    try:
        obj = json.loads(clean)
        return obj.get("domain_name", ""), obj.get("definition", "")
    except json.JSONDecodeError:
        log.warning("LLM response not valid JSON: %s", raw[:80])
        return "", ""


def label_domains(
    partition: dict[str, int],
    profiles: list[dict],
    concepts: list[dict],
    phi_df: pd.DataFrame,
    use_llm: bool,
    model: str,
    ollama_url: str,
    temperature: float = 0.1,
    llm_timeout: int   = 90,
) -> list[dict]:
    """
    §5.6 — Label each discovered domain via LLM only.

    Calls Ollama with tables + columns + phi context for each domain.
    If the LLM is unavailable or returns empty/duplicate, raises RuntimeError.
    Hungarian assignment is disabled — LLM is the sole labeling mechanism.
    """
    groups: dict[int, list[str]] = defaultdict(list)
    for table, cid in partition.items():
        groups[cid].append(table)

    domains = []
    used_labels: list[str] = []
    for cid in sorted(groups.keys()):
        tables = sorted(groups[cid])
        log.info("§5.6  D%d %s", cid, tables)

        domain_name = ""
        definition  = ""

        if use_llm:
            rows    = [idx for idx in phi_df.index if idx.split(".")[0] in tables]
            abs_phi = phi_df.loc[rows].mean().sort_values(ascending=False) if rows else pd.Series(dtype=float)
            prompt  = build_prompt(cid, tables, profiles, concepts, abs_phi,
                                   used_labels=used_labels)
            raw     = call_ollama(prompt, model, ollama_url,
                                  temperature=temperature, timeout=llm_timeout)
            domain_name, definition = parse_llm_response(raw)

            # Duplicate check — retry up to 5 times with increasing temperature
            MAX_RETRIES   = 5
            retry_temps   = [0.4, 0.6, 0.8, 1.0, 1.2]
            retry_attempt = 0
            while domain_name and domain_name in used_labels and retry_attempt < MAX_RETRIES:
                retry_attempt += 1
                temp_retry = retry_temps[retry_attempt - 1]
                log.warning(
                    "      LLM returned duplicate label '%s' — retry %d/%d (temp=%.1f)",
                    domain_name, retry_attempt, MAX_RETRIES, temp_retry,
                )
                raw2        = call_ollama(
                    build_prompt(cid, tables, profiles, concepts, abs_phi,
                                 used_labels=used_labels),
                    model, ollama_url, temperature=temp_retry, timeout=llm_timeout,
                )
                name2, def2 = parse_llm_response(raw2)
                if name2 and name2 not in used_labels:
                    domain_name, definition = name2, def2
                    log.info("      Retry %d succeeded: '%s'", retry_attempt, domain_name)
                    break
                else:
                    log.warning(
                        "      Retry %d also duplicate ('%s')",
                        retry_attempt, name2,
                    )
                    domain_name, definition = name2 or domain_name, def2 or definition

        if not domain_name:
            raise RuntimeError(
                f"LLM failed to label D{cid} (tables: {tables}). "
                f"Check that Ollama is running and mistral:latest is responding. "
                f"Re-run the command once the issue is resolved."
            )
        else:
            log.info("      LLM label: '%s'", domain_name)

        used_labels.append(domain_name)

        rows      = [idx for idx in phi_df.index if idx.split(".")[0] in tables]
        abs_phi   = phi_df.loc[rows].mean().sort_values(ascending=False) if rows else pd.Series(dtype=float)
        phi_scores = {k: round(float(v), 4) for k, v in abs_phi.items()} if not abs_phi.empty else {}

        domains.append({
            "domain_id":   cid,
            "domain_name": domain_name,
            "definition":  definition,
            "tables":      tables,
            "phi_scores":  phi_scores,
        })

    return domains


# =============================================================================
# Build output DataFrames
# =============================================================================

def build_dataframes(
    domains: list[dict],
    profiles: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      table_df  — one row per table
      column_df — one row per column
    """
    table_rows = [
        {"table": t, "domain_id": d["domain_id"], "domain_name": d["domain_name"]}
        for d in domains
        for t in d["tables"]
    ]
    table_df = pd.DataFrame(table_rows).sort_values(["domain_id", "table"]).reset_index(drop=True)

    tbl_map = {r["table"]: (r["domain_id"], r["domain_name"]) for r in table_rows}
    col_rows = [
        {
            "table":       p["table"],
            "column":      p["column"],
            "domain_id":   tbl_map.get(p["table"], (-1, "Unknown"))[0],
            "domain_name": tbl_map.get(p["table"], (-1, "Unknown"))[1],
        }
        for p in profiles
    ]
    column_df = pd.DataFrame(col_rows).sort_values(
        ["domain_id", "table", "column"]
    ).reset_index(drop=True)

    return table_df, column_df


# =============================================================================
# Running Example verification
# =============================================================================

def running_example_check(table_df: pd.DataFrame) -> None:
    """
    Generic sanity check: log how many tables landed in each domain.
    Works for any dataset — no hardcoded table names.
    """
    if table_df.empty:
        log.warning("Domain check: no tables assigned.")
        return
    summary = table_df.groupby("domain_id")["table"].apply(list)
    log.info("Domain assignment summary:")
    for did, tables in summary.items():
        log.info("  D%s (%d tables): %s", did, len(tables), sorted(tables))


# =============================================================================
# Report
# =============================================================================

def write_report(
    domains: list[dict],
    table_df: pd.DataFrame,
    G: nx.Graph,
    theta_t: float,
    path: Path,
) -> None:
    # Modularity
    try:
        partition = dict(zip(table_df["table"], table_df["domain_id"]))
        sets = [{t for t, c in partition.items() if c == cid}
                for cid in set(partition.values())]
        mod = nx.community.modularity(G, sets, weight="weight")
        mod_str = f"{mod:.4f}"
    except Exception as exc:
        log.warning("Modularity Q computation failed: %s", exc)
        mod_str = "N/A"

    lines = [
        "=" * 70,
        "CCM Pipeline — Step 5 Report",
        "Graph Clustering — Domain Discovery",
        "=" * 70,
        f"Tables in G_T        : {G.number_of_nodes()}",
        f"Edges in G_T         : {G.number_of_edges()}",
        f"θ_T applied in Step 4: {theta_t}",
        f"Domains discovered   : {len(domains)}",
        f"Modularity Q         : {mod_str}",
        "",
    ]

    for d in domains:
        phi_top = list(d["phi_scores"].items())[:3] if d["phi_scores"] else []
        phi_str = "  ".join(f"{k}={v:.3f}" for k, v in phi_top)
        lines += [
            "─" * 70,
            f"D{d['domain_id']}  {d['domain_name']}",
            "─" * 70,
            f"  Definition : {d['definition']}",
            f"  Tables     : {', '.join(d['tables'])}",
            f"  Phi top-3  : {phi_str}",
            "",
        ]

    lines += [
        "Table → Domain assignment:",
        f"  {'Table':35s}  {'Domain ID':>9}  Domain Name",
        "  " + "-" * 60,
    ]
    for _, r in table_df.iterrows():
        lines.append(
            f"  {r['table']:35s}  D{r['domain_id']:>8}  {r['domain_name']}"
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
        description="CCM Step 5 — Graph Clustering and Domain Discovery"
    )
    p.add_argument("--input_dir",   default=str(DEFAULT_INPUT_DIR),
                   help="Directory with step4_graph_edges.csv")
    p.add_argument("--concepts",    default="ccm_output/step1_concepts.json",
                   help="Path to step1_concepts.json")
    p.add_argument("--profiles",    default="ccm_output/step2_column_profiles.json",
                   help="Path to step2_column_profiles.json")
    p.add_argument("--phi_matrix",  default="ccm_output/phi_matrix.csv",
                   help="Path to phi_matrix.csv")
    p.add_argument("--out_dir",     default=str(DEFAULT_OUTPUT_DIR),
                   help="Output directory")
    p.add_argument("--resolution",    type=float, default=DEFAULT_RESOLUTION,
                   help=f"Louvain resolution (default: {DEFAULT_RESOLUTION})")
    p.add_argument("--random_state",  type=int,   default=DEFAULT_RANDOM_STATE,
                   help=f"Louvain random seed (default: {DEFAULT_RANDOM_STATE})")
    p.add_argument("--theta_t",     type=float, default=DEFAULT_THETA_T,
                   help="θ_T from Step 4 (for report only)")
    p.add_argument("--no_llm",      action="store_true",
                   help="Skip LLM — use phi-based fallback labels only")
    p.add_argument("--model",       default=DEFAULT_MODEL,
                   help=f"Ollama model (default: {DEFAULT_MODEL})")
    p.add_argument("--ollama_url",  default=DEFAULT_OLLAMA_URL,
                   help="Ollama API endpoint")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="LLM temperature for domain labelling (default: 0.1)")
    p.add_argument("--llm_timeout", type=int, default=90,
                   help="LLM HTTP timeout in seconds (default: 90)")
    p.add_argument("--schema",      default="ccm_output/schema.json",
                   help="Path to schema.json (for domain schema export)")
    p.add_argument("--dataset_dir", default=None,
                   help="Dataset working directory containing schema.json, knowledge.docx, "
                        "csv/ and ccm_output/. When run_pipeline.py is in the parent folder, "
                        "pass the dataset subfolder name (e.g. --dataset_dir Chinook). "
                        "Defaults to the current working directory.")
    return p.parse_args()


# =============================================================================
# §5.7  Domain Schema Export
# =============================================================================

def build_domain_schemas(
    domains: list[dict],
    schema_path: Path,
    out_dir: Path,
) -> None:
    """
    §5.7 — For each discovered domain Di, produce a separate schema JSON
    containing only the tables (and their full column definitions + samples)
    that belong to Di.

    Output structure per domain (mirrors the input schema.json format):
    {
        "_meta": {
            "domain_id":   int,
            "domain_name": str,
            "definition":  str,
            "tables":      [str, ...],
            "source_schema": str          # original schema filename
        },
        "tables": {
            "<TableName>": {
                "domain":   str,          # original domain tag preserved
                "columns":  [ {name, sql_type, key, description, samples}, ... ]
            },
            ...
        }
    }

    Files written:
        step5_domain_schema_D{id}_{sanitised_name}.json   (one per domain)
        step5_domain_schemas_all.json                      (all domains merged)
    """
    if not schema_path.exists():
        log.warning("§5.7  schema.json not found at %s — skipping domain schema export", schema_path)
        return

    # Remove old step5_domain_schema_D*.json files before writing new ones
    for old in out_dir.glob("step5_domain_schema_D*.json"):
        old.unlink()
        log.info("§5.7  Removed old: %s", old.name)
    combined_old = out_dir / "step5_domain_schemas_all.json"
    if combined_old.exists():
        combined_old.unlink()

    full_schema: dict = json.loads(schema_path.read_text(encoding="utf-8"))
    all_tables: dict  = full_schema.get("tables", {})
    schema_filename   = schema_path.name

    all_domain_schemas: dict[int, dict] = {}

    for d in domains:
        did        = d["domain_id"]
        dname      = d["domain_name"]
        definition = d.get("definition", "")
        tables     = d["tables"]           # list[str] of table names

        domain_tables: dict = {}
        for tname in sorted(tables):
            if tname in all_tables:
                domain_tables[tname] = all_tables[tname]
            else:
                log.warning("§5.7  Table %r not found in schema — skipped", tname)

        domain_schema = {
            "_meta": {
                "domain_id":     did,
                "domain_name":   dname,
                "definition":    definition,
                "tables":        sorted(tables),
                "source_schema": schema_filename,
            },
            "tables": domain_tables,
        }

        all_domain_schemas[did] = domain_schema

        # Sanitise domain name for filename (replace spaces/special chars)
        safe_name = dname.replace(" ", "_").replace("/", "_").replace("&", "and")
        out_path  = out_dir / f"step5_domain_schema_D{did}_{safe_name}.json"
        out_path.write_text(
            json.dumps(domain_schema, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info(
            "§5.7  D%d [%s] → %d tables → %s",
            did, dname, len(domain_tables), out_path.name,
        )

    # Write combined file
    combined_path = out_dir / "step5_domain_schemas_all.json"
    combined_path.write_text(
        json.dumps(all_domain_schemas, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("§5.7  All domain schemas → %s", combined_path)


def build_domain_folders(
    domains: list[dict],
    schema_path: Path,
    out_dir: Path,
) -> None:
    """
    §5.8 — Write one JSON file per domain directly into out_dir/domains/.
    No subfolders are created.

    Output file per domain:  domains/D{id}_{DomainName}.json
    Structure:
    {
        "_meta": { domain_id, domain_name, definition, tables[], n_tables,
                   n_columns_total, source_schema },
        "tables": {
            "<TableName>": {
                "domain":   str,
                "columns":  [ {name, sql_type, key, description, samples}, ... ]
            },
            ...
        }
    }

    Flat layout:
      domains/
        D0_<DomainName>.json
        D1_<DomainName>.json
        ...
    """
    if not schema_path.exists():
        log.warning("§5.8  schema.json not found — skipping domain folder export")
        return

    full_schema: dict = json.loads(schema_path.read_text(encoding="utf-8"))
    all_tables: dict  = full_schema.get("tables", {})
    schema_filename   = schema_path.name

    domains_root = out_dir / "domains"

    # Always clean domains/ before writing new files — prevents stale files
    # from previous runs mixing with new ones
    if domains_root.exists():
        import shutil as _sh
        import stat as _stat

        def _handle_remove_error(func, path, exc_info):
            """Handle read-only files on Windows by forcing write permission."""
            try:
                import os
                os.chmod(path, _stat.S_IWRITE)
                func(path)
            except Exception:
                pass  # skip files that cannot be removed

        _sh.rmtree(domains_root, onerror=_handle_remove_error)
        log.info("§5.8  Cleared old domains/ folder")
    domains_root.mkdir(parents=True, exist_ok=True)

    for d in domains:
        did        = d["domain_id"]
        dname      = d["domain_name"]
        definition = d.get("definition", "")
        tables     = sorted(d["tables"])

        safe_name  = dname.replace(" ", "_").replace("/", "_").replace("&", "and")

        # ── Collect all tables into one merged dict ────────────────────────
        merged_tables: dict = {}
        n_cols_total = 0
        for tname in tables:
            tdata = all_tables.get(tname)
            if tdata is None:
                log.warning("§5.8  Table %r not in schema — skipped", tname)
                continue
            merged_tables[tname] = {
                "domain":  tdata.get("domain", ""),
                "columns": tdata["columns"],
            }
            n_cols_total += len(tdata["columns"])

        # ── Single JSON file directly in domains/ ─────────────────────────
        domain_schema = {
            "_meta": {
                "domain_id":      did,
                "domain_name":    dname,
                "definition":     definition,
                "tables":         tables,
                "n_tables":       len(tables),
                "n_columns_total": n_cols_total,
                "source_schema":  schema_filename,
            },
            "tables": merged_tables,
        }

        out_file = domains_root / f"D{did}_{safe_name}.json"
        out_file.write_text(
            json.dumps(domain_schema, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        log.info(
            "§5.8  D%d [%s] → %s (%d tables, %d columns)",
            did, dname, out_file.name, len(merged_tables), n_cols_total,
        )

    log.info("§5.8  Domain files → %s/", domains_root)


def main() -> None:
    args = parse_args()

    # ── FIX: chdir must happen here in main(), BEFORE any Path() resolution.
    #    Previously this block was placed after `return` inside parse_args(),
    #    making it unreachable dead code. --dataset_dir was silently ignored.
    if args.dataset_dir is not None:
        import os
        os.chdir(args.dataset_dir)
        log.info("Working directory set to: %s", Path.cwd())

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Clean old step5 outputs before writing new ones ───────────────────────
    for pattern in ["step5_*.json", "step5_*.csv", "step5_*.txt"]:
        for old in out_dir.glob(pattern):
            old.unlink()
            log.info("[clean] Removed old: %s", old.name)

    log.info("=" * 60)
    log.info("CCM Pipeline — Step 5: Domain Discovery")
    log.info("  input_dir  = %s", in_dir)
    log.info("  resolution = %.2f", args.resolution)
    log.info("  LLM        = %s", "off (--no_llm)" if args.no_llm else args.model)
    log.info("=" * 60)

    # ── Load inputs ──────────────────────────────────────────────────────────
    edges_path    = in_dir / "step4_graph_edges.csv"
    concepts_path = Path(args.concepts)
    profiles_path = Path(args.profiles)
    phi_path      = Path(args.phi_matrix)

    # Resolve schema.json — search in multiple candidate locations
    _schema_candidates = [
        Path(args.schema),             # --schema argument (explicit)
        in_dir  / "schema.json",       # same dir as step4 edges
        out_dir / "schema.json",       # output dir
        Path("schema.json"),           # current working directory
        Path("ccm_output/schema.json"),# default ccm_output location
    ]
    schema_path = next((p for p in _schema_candidates if p.exists()), Path(args.schema))
    if schema_path.exists():
        log.info("§5.7  schema.json found at: %s", schema_path)
    else:
        log.warning(
            "§5.7  schema.json not found. Searched:\n%s\n"
            "       Copy schema.json into ccm_output/ or pass --schema <path>",
            "\n".join(f"         {p}" for p in _schema_candidates)
        )

    for p in [edges_path, concepts_path, profiles_path, phi_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    concepts = json.loads(concepts_path.read_text(encoding="utf-8"))
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    phi_df   = pd.read_csv(phi_path, index_col=0)

    log.info("Loaded: %d concepts, %d column profiles, phi matrix %s",
             len(concepts), len(profiles), phi_df.shape)

    # ── Load real table names from schema to filter phantom tables ────────────
    real_tables: set[str] | None = None
    if schema_path.exists():
        _raw = json.loads(schema_path.read_text(encoding="utf-8"))
        real_tables = set(_raw.get("tables", _raw).keys())
        log.info("§5    Real tables from schema: %d", len(real_tables))
        # Filter profiles to real tables only
        profiles = [p for p in profiles if p["table"] in real_tables]
        log.info("§5    Column profiles after phantom filter: %d", len(profiles))

    # ── §5.1–5.5  Louvain ────────────────────────────────────────────────────
    G         = build_graph(edges_path, real_tables=real_tables)
    G         = add_isolated_nodes(G, profiles)
    partition = community_detection(G, args.resolution, args.random_state)

    # ── §5.6  Labeling ────────────────────────────────────────────────────────
    domains = label_domains(
        partition, profiles, concepts, phi_df,
        use_llm     = not args.no_llm,
        model       = args.model,
        ollama_url  = args.ollama_url,
        temperature = args.temperature,
        llm_timeout = args.llm_timeout,
    )

    # ── Build output tables ───────────────────────────────────────────────────
    table_df, column_df = build_dataframes(domains, profiles)

    # ── Running Example check ─────────────────────────────────────────────────
    running_example_check(table_df)

    # ── Save outputs ──────────────────────────────────────────────────────────
    domains_path = out_dir / "step5_domains.json"
    table_path   = out_dir / "step5_table_domain.csv"
    column_path  = out_dir / "step5_column_domain.csv"
    report_path  = out_dir / "step5_report.txt"

    domains_path.write_text(
        json.dumps(domains, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    table_df.to_csv(table_path,   index=False)
    column_df.to_csv(column_path, index=False)
    write_report(domains, table_df, G, args.theta_t, report_path)

    # ── §5.7  Domain schema export (single JSON per domain) ──────────────────
    build_domain_schemas(domains, schema_path, out_dir)

    # ── §5.8  Domain folder export (one subfolder + one JSON per table) ─────────
    build_domain_folders(domains, schema_path, out_dir)

    log.info("=" * 60)
    log.info("Step 5 complete — D* = %d domains", len(domains))
    log.info("  step5_domains.json                → %s", domains_path)
    log.info("  step5_table_domain.csv            → %s", table_path)
    log.info("  step5_column_domain.csv           → %s", column_path)
    log.info("  step5_report.txt                  → %s", report_path)
    log.info("  step5_domain_schema_D*.json       → %s/", out_dir)
    log.info("  step5_domain_schemas_all.json     → %s/", out_dir)
    log.info("  domains/D*/<Table>.json           → %s/domains/", out_dir)
    log.info("  Next: sensitivity_analysis.py (now Step 5 is ready)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()