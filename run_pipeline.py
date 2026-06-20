"""
run_pipeline.py
===============
Runs the full CCM TwoLevelDomainDiscovery pipeline sequentially:

  Step 1+2  1_knowledge_concept_embedding.py  — concept extraction + column profiling
  Step 3a   2_p_stat_name_sem.py              — P_stat, P_name, P_sem computation
  Step 3b   3_sim_attr_weights.py             — derive weights w1, w2, w3
  Step 3c   4_column_graph.py                 — Sim_attr + column graph G_A
  Step 4    5_table_similarity.py             — table similarity + graph G_T
  Step 5    6_domain_discovery.py             — Louvain clustering + domain labelling

Usage:
    # Minimal — uses all defaults (schema.json + ./ccm_output/)
    python run_pipeline.py

    # With knowledge file (recommended)
    python run_pipeline.py --schema schema.json --knowledge knowledge.docx

    # Resume from a specific step after a failure
    python run_pipeline.py --schema schema.json --knowledge knowledge.docx --start_from step3b

    # Dry run — print commands without executing
    python run_pipeline.py --dry_run

    # Full control
    python run_pipeline.py ^
        --schema    schema.json ^
        --out_dir   ccm_output ^
        --knowledge knowledge.docx ^
        --theta_a   0.60 ^
        --theta_t   0.60 ^
        --model     mistral-ctx4k

FIXES in this version:
  - out_dir is always kept as a RELATIVE path — never resolved to absolute
  - Step 3b now receives --input_dir and --out_dir explicitly
  - --model default changed to mistral-ctx4k
  - --embed_model argument name made consistent (underscore everywhere)
  - knowledge path is resolved relative to script location, not cwd
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    from pipeline_utils import setup_log_file, clean_run_pipeline, make_run_tag
    _UTILS = True
except ImportError:
    _UTILS = False

    def make_run_tag(theta_a: float, theta_t: float, resolution: float) -> str:
        return f"tA{theta_a}_tT{theta_t}_r{resolution}"

log = logging.getLogger(__name__)


# =============================================================================
# Script filenames — tries each candidate, uses first one found
# =============================================================================

SCRIPT_CANDIDATES = {
    "step12": [
        "1_knowledge_concept_embedding.py",
        "1 -knowledge_concept_embedding.py",
        "1_-knowledge_concept_embedding.py",
        "knowledge_concept_embedding.py",
    ],
    "step3a": [
        "2_p_stat_name_sem.py",
        "2 - p_stat_name_sem.py",
        "2_-_p_stat_name_sem.py",
        "p_stat_name_sem.py",
    ],
    "step3b": [
        "3_sim_attr_weights.py",
        "3 - sim_attr_weights.py",
        "3_-_sim_attr_weights.py",
        "sim_attr_weights.py",
    ],
    "step3c": [
        "4_column_graph.py",
        "4 - column_graph.py",
        "4_-_column_graph.py",
        "column_graph.py",
    ],
    "step4": [
        "5_table_similarity.py",
        "5 - table_similarity.py",
        "5_-_table_similarity.py",
        "table_similarity.py",
    ],
    "step5": [
        "6_domain_discovery.py",
        "6 - domain_discovery.py",
        "6_-_domain_discovery.py",
        "domain_discovery.py",
    ],
}

STEP_LABELS = {
    "step12": "Steps 1+2  — Knowledge Extraction + Column Profiling",
    "step3a": "Step 3a   — P_stat · P_name · P_sem",
    "step3b": "Step 3b   — Derive Weights w1 w2 w3",
    "step3c": "Step 3c   — Sim_attr + Column Graph G_A",
    "step4":  "Step 4    — Table Similarity + Graph G_T",
    "step5":  "Step 5    — Domain Discovery (Louvain + Labelling)",
}

STEP_ORDER = ["step12", "step3a", "step3b", "step3c", "step4", "step5"]


# =============================================================================
# Script resolution
# =============================================================================

def resolve_script(step_key: str, script_dir: Path) -> Path | None:
    """Find the actual filename for a step by trying all candidate names."""
    for candidate in SCRIPT_CANDIDATES[step_key]:
        p = script_dir / candidate
        if p.exists():
            return p
    return None


# =============================================================================
# Build commands
# =============================================================================

def build_commands(args, script_dir: Path, out_dir_str: str,
                   schema_str: str, run_out_dir_str: str | None = None) -> dict:
    """
    Return ordered dict of step_key -> [python, script, arg1, arg2, ...]

    Steps 1+2, 3a, 3b  →  out_dir_str       (shared, parameter-independent)
    Steps 3c, 4, 5     →  run_out_dir_str   (per-run, encodes theta_A/T + resolution)

    If run_out_dir_str is None (legacy / full-clean run), all steps use out_dir_str.
    """
    if run_out_dir_str is None:
        run_out_dir_str = out_dir_str
    python   = sys.executable
    theta_a  = str(args.theta_a)
    theta_t  = str(args.theta_t)
    model    = args.model
    ollama       = args.ollama_url          # base URL for script 1 (e.g. http://localhost:11434)
    # script 6 (6_domain_discovery.py) calls /api/generate directly
    ollama_gen   = ollama.rstrip("/") + "/api/generate" if not ollama.endswith("/api/generate") else ollama
    res      = str(args.resolution)
    seed     = str(args.random_state)

    def script(key):
        p = resolve_script(key, script_dir)
        if p is None:
            raise FileNotFoundError(
                f"Cannot find script for {key}. Tried:\n" +
                "\n".join(f"  {script_dir / c}"
                          for c in SCRIPT_CANDIDATES[key])
            )
        return str(p)

    cmds = {}

    # ── Step 1+2 ──────────────────────────────────────────────────────────────
    cmd12 = [python, script("step12"),
             "--schema", schema_str,
             "--out",    out_dir_str]
    if args.embed_model:
        cmd12 += ["--embed_model", args.embed_model]
    if args.knowledge:
        cmd12 += ["--knowledge", args.knowledge]
    # LLM params for concept extraction
    cmd12 += ["--llm_backend",  args.llm_backend,
              "--ollama_model", model,
              "--ollama_url",   ollama,
              "--num_predict",  str(args.num_predict),
              "--num_ctx",      str(args.num_ctx),
              "--temperature",  str(args.temperature),
              "--llm_timeout",  str(args.llm_timeout)]
    if args.hf_token:
        cmd12 += ["--hf_token", args.hf_token]
    if args.hf_model:
        cmd12 += ["--hf_model", args.hf_model]
    if args.sample_size:
        cmd12 += ["--sample_size", str(args.sample_size)]
    cmds["step12"] = cmd12

    # ── Step 3a ───────────────────────────────────────────────────────────────
    cmds["step3a"] = [python, script("step3a"),
                      "--input-dir", out_dir_str,
                      "--out-dir",   out_dir_str]

    # ── Step 3b ───────────────────────────────────────────────────────────────
    cmds["step3b"] = [python, script("step3b"),
                      "--input_dir", out_dir_str,
                      "--out_dir",   out_dir_str]

    # ── Step 3c ───────────────────────────────────────────────────────────────
    # Steps 3c, 4, 5 write to the per-run subfolder (encodes theta_A/T + resolution)
    # Step 3c reads shared inputs from out_dir_str (steps 3a/3b outputs live there)
    cmds["step3c"] = [python, script("step3c"),
                      "--input-dir", out_dir_str,
                      "--out-dir",   run_out_dir_str,
                      "--theta",     theta_a]

    # ── Step 4 ────────────────────────────────────────────────────────────────
    cmds["step4"] = [python, script("step4"),
                     "--input_dir", run_out_dir_str,
                     "--schema",    schema_str,
                     "--out_dir",   run_out_dir_str,
                     "--theta_t",   theta_t]

    # ── Step 5 ────────────────────────────────────────────────────────────────
    # Concepts, profiles, phi_matrix are shared (written by steps 1+2 to out_dir_str)
    cmd5 = [python, script("step5"),
            "--input_dir",    run_out_dir_str,
            "--concepts",     str(Path(out_dir_str) / "step1_concepts.json"),
            "--profiles",     str(Path(out_dir_str) / "step2_column_profiles.json"),
            "--phi_matrix",   str(Path(out_dir_str) / "phi_matrix.csv"),
            "--schema",       schema_str,
            "--out_dir",      run_out_dir_str,
            "--resolution",   res,
            "--random_state", seed,
            "--theta_t",      theta_t,
            "--model",        model,
            "--ollama_url",   ollama_gen,
            "--temperature",  str(args.temperature),
            "--llm_timeout",  str(args.llm_timeout)]
    if args.no_llm:
        cmd5.append("--no_llm")
    cmds["step5"] = cmd5

    return cmds


# =============================================================================
# Run one step
# =============================================================================

def run_step(step_key: str, cmd: list, dry_run: bool,
             cwd: str | None = None) -> bool:
    """Execute one pipeline step. Returns True on success."""
    label = STEP_LABELS[step_key]
    print()
    print("-" * 70)
    print(f"  {label}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    print("-" * 70)

    if dry_run:
        print("  [DRY RUN] Command not executed.")
        return True

    t0     = time.time()
    result = subprocess.run(cmd, text=True, cwd=cwd)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  OK  Completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n    FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline(args) -> None:

    # ── Resolve paths ─────────────────────────────────────────────────────────
    # script_dir: where the 6 pipeline scripts live — defaults to run_pipeline.py's own folder
    if args.script_dir is not None:
        script_dir = Path(args.script_dir).resolve()
    else:
        script_dir = Path(__file__).resolve().parent

    # dataset_dir: working directory for schema.json, knowledge.docx, csv/, ccm_output/
    # All subprocess steps run with cwd=dataset_dir so relative paths resolve correctly
    if args.dataset_dir is not None:
        dataset_dir = Path(args.dataset_dir).resolve()
    else:
        dataset_dir = Path.cwd()

    out_dir_path = Path(args.out_dir)
    out_dir_abs  = (dataset_dir / out_dir_path).resolve()
    out_dir_str  = str(out_dir_path)          # kept relative — resolved by cwd in subprocess
    schema_path  = Path(args.schema)
    schema_abs   = (dataset_dir / schema_path).resolve()
    schema_str   = str(schema_path)           # kept relative — resolved by cwd in subprocess

    # ── Per-run subfolder (encodes theta_A, theta_T, resolution) ─────────────
    # Steps 1+2, 3a, 3b  →  ccm_output/           (shared, parameter-independent)
    # Steps 3c, 4, 5     →  ccm_output/<run_tag>/  (parameter-specific)
    run_tag         = make_run_tag(args.theta_a, args.theta_t, args.resolution)
    run_out_dir_abs = out_dir_abs / run_tag
    run_out_dir_str = str(out_dir_path / run_tag)   # relative, resolved by cwd in subprocess

    # ── Create output folders ─────────────────────────────────────────────────
    out_dir_abs.mkdir(parents=True, exist_ok=True)
    run_out_dir_abs.mkdir(parents=True, exist_ok=True)
    logs_dir = dataset_dir / args.log_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging — console + file ──────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"run_pipeline_{ts}.log"
    fh       = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)
    log.info("Log file: %s", log_path)

    # ── Cleanup previous outputs ──────────────────────────────────────────────
    # --clean without --start_from: remove only this run's subfolder
    # (shared Steps 1-3b outputs in ccm_output/ are preserved)
    if args.clean and not args.dry_run and not args.start_from:
        if _UTILS:
            clean_run_pipeline(out_dir_abs, run_tag=run_tag)
        else:
            import shutil as _sh
            if run_out_dir_abs.exists():
                _sh.rmtree(run_out_dir_abs)
                log.info("[clean] Removed run dir %s", run_out_dir_abs)
            run_out_dir_abs.mkdir(parents=True, exist_ok=True)

    # ── Validate scripts exist ────────────────────────────────────────────────
    missing = []
    found   = {}
    for key in SCRIPT_CANDIDATES:
        p = resolve_script(key, script_dir)
        if p is None:
            missing.append(key)
        else:
            found[key] = p

    if found:
        print("\n  Scripts found:")
        for key, p in found.items():
            print(f"    {STEP_LABELS[key][:45]:45s}  {p.name}")

    if missing and not args.dry_run:
        print("\n[ERROR] The following pipeline scripts were NOT found:")
        for key in missing:
            print(f"\n  {STEP_LABELS[key]}:")
            for c in SCRIPT_CANDIDATES[key]:
                print(f"    tried: {script_dir / c}")
        print("\nEnsure run_pipeline.py is in the same folder as the 6 scripts,")
        print("or use --script_dir to point to the correct folder.")
        sys.exit(1)

    # ── Validate schema ───────────────────────────────────────────────────────
    schema_abs = (dataset_dir / Path(args.schema)).resolve()
    if not schema_abs.exists() and not args.dry_run:
        print(f"\n[ERROR] Schema file not found: {schema_abs}")
        print("Pass --schema <path/to/schema.json>")
        sys.exit(1)

    # ── Validate knowledge file ───────────────────────────────────────────────
    if args.knowledge:
        knowledge_abs = (dataset_dir / args.knowledge).resolve()
        if not knowledge_abs.exists() and not args.dry_run:
            print(f"\n[ERROR] Knowledge file not found: {knowledge_abs}")
            print("Pass --knowledge <path/to/knowledge.docx>")
            sys.exit(1)

    # ── Build commands ────────────────────────────────────────────────────────
    cmds = build_commands(args, script_dir, out_dir_str, schema_str,
                          run_out_dir_str=run_out_dir_str)

    # ── Banner — print and log ────────────────────────────────────────────────
    banner_lines = [
        "",
        "=" * 70,
        "  CCM Pipeline — TwoLevelDomainDiscovery",
        f"  Started    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Dataset dir: {dataset_dir}",
        f"  Script dir : {script_dir}",
        f"  Schema     : {schema_abs}",
        f"  Shared dir : {out_dir_abs}   (Steps 1+2, 3a, 3b)",
        f"  Run dir    : {run_out_dir_abs}   (Steps 3c, 4, 5)",
        f"  Run tag    : {run_tag}",
        f"  Log file   : {log_path}",
        f"  Knowledge  : {args.knowledge if args.knowledge else 'K_RAW in script 1'}",
        f"  theta_A    : {args.theta_a}  theta_T: {args.theta_t}",
        f"  LLM model  : {'OFF (--no_llm)' if args.no_llm else args.model}  backend={args.llm_backend}",
        f"  ollama_url : {args.ollama_url}",
        f"  num_predict: {args.num_predict}  num_ctx={args.num_ctx}  temperature={args.temperature}  timeout={args.llm_timeout}s",
        f"  embed_model: {args.embed_model}",
        f"  resolution : {args.resolution}  random_state={args.random_state}",
        f"  Clean run  : {getattr(args, 'clean', False)}",
        f"  Dry run    : {args.dry_run}",
        "=" * 70,
    ]
    for line in banner_lines:
        print(line)
        log.info(line)

    # ── Determine steps to run ────────────────────────────────────────────────
    steps = list(STEP_ORDER)
    if args.start_from:
        if args.start_from not in steps:
            print(f"[ERROR] --start_from must be one of: {steps}")
            sys.exit(1)
        idx   = steps.index(args.start_from)
        steps = steps[idx:]
        print(f"\n  Resuming from: {STEP_LABELS[args.start_from]}")

    # ── Execute ───────────────────────────────────────────────────────────────
    pipeline_start = time.time()
    failed_at      = None

    for step_key in steps:
        ok = run_step(step_key, cmds[step_key], dry_run=args.dry_run,
                      cwd=str(dataset_dir))
        if not ok:
            failed_at = step_key
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    print()
    print("=" * 70)
    if failed_at:
        print(f"  PIPELINE FAILED at: {STEP_LABELS[failed_at]}")
        print(f"  Fix the error above and re-run from that step:")
        print(f"    python run_pipeline.py --start_from {failed_at} [other args]")
    else:
        print("  PIPELINE COMPLETE ")
        print(f"  Total time : {total:.1f}s  ({total/60:.1f} min)")
        print(f"  Run dir    : {run_out_dir_abs}")
        print(f"  Run tag    : {run_tag}")
        print()
        print("  Key output files:")
        outputs = [
            ("step3_graph_edges.csv",            "G_A column graph edges"),
            ("step4_graph_edges.csv",            "G_T table graph edges"),
            ("step5_domains.json",               "Discovered domain partition D*"),
            ("step5_table_domain.csv",           "Table → domain assignment"),
            ("step5_column_domain.csv",          "Column → domain assignment"),
            ("step5_report.txt",                 "Full pipeline report"),
        ]
        for fname, desc in outputs:
            p = run_out_dir_abs / fname
            mark = "+" if p.exists() else "."
            print(f"    {mark}  {fname:40s}  {desc}")
    print("=" * 70)

    if failed_at:
        sys.exit(1)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full CCM TwoLevelDomainDiscovery pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard run
  python run_pipeline.py --schema schema.json --knowledge knowledge.docx

  # Groq-generated knowledge, Ollama for pipeline
  python run_pipeline.py --schema schema.json --knowledge knowledge.docx

  # Custom model and timeouts
  python run_pipeline.py --schema schema.json --knowledge knowledge.docx ^
      --model mistral-ctx4k --num_predict 1024 --llm_timeout 1800

  # Resume after failure
  python run_pipeline.py --schema schema.json --knowledge knowledge.docx ^
      --start_from step3b

  # Dry run — just print commands
  python run_pipeline.py --dry_run
"""
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Paths")
    g.add_argument("--schema",     default="schema.json",
                   help="Path to schema.json relative to --dataset_dir (default: schema.json)")
    g.add_argument("--out_dir",    default="ccm_output",
                   help="Output directory relative to --dataset_dir (default: ccm_output)")
    g.add_argument("--script_dir", default=None,
                   help="Directory containing the 6 pipeline scripts. "
                        "Defaults to the directory of run_pipeline.py itself.")
    g.add_argument("--dataset_dir", default=None,
                   help="Dataset working directory (schema.json, knowledge.docx, csv/, ccm_output/). "
                        "Defaults to current working directory. "
                        "Use when run_pipeline.py lives in a parent folder: "
                        "--dataset_dir Chinook")
    g.add_argument("--knowledge",  default=None,
                   help="Path to knowledge file (.docx or .txt), relative to --dataset_dir.")

    # ── Thresholds ────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Thresholds")
    g.add_argument("--theta_a", type=float, default=0.65,
                   help="Column graph edge threshold θ_A (default: 0.65, fixed globally)")
    g.add_argument("--theta_t", type=float, default=0.60,
                   help="Table graph edge threshold θ_T — tune per dataset (default: 0.60)")

    # ── LLM backend ───────────────────────────────────────────────────────────
    g = parser.add_argument_group("LLM backend")
    g.add_argument("--no_llm", action="store_true",
                   help="Skip LLM — use phi-based fallback labels only")
    g.add_argument("--llm_backend", default="ollama",
                   choices=["ollama", "huggingface"],
                   help="LLM backend for concept extraction in Script 1 "
                        "(default: ollama)")
    g.add_argument("--model", default="mistral-ctx4k",
                   help="Ollama model name used in Script 1 + Script 6 "
                        "(default: mistral-ctx4k)")
    g.add_argument("--ollama_url", default="http://localhost:11434",
                   help="Ollama base URL (default: http://localhost:11434)")
    g.add_argument("--hf_token", default=None,
                   help="HuggingFace API token (for --llm_backend huggingface). "
                        "Also via HF_TOKEN env var.")
    g.add_argument("--hf_model", default=None,
                   help="HuggingFace model ID (default: mistralai/Mistral-7B-Instruct-v0.2)")

    # ── LLM generation constraints ─────────────────────────────────────────────
    g = parser.add_argument_group(
        "LLM generation constraints",
        "Tune these per model without editing individual scripts."
    )
    g.add_argument("--num_predict", type=int, default=512,
                   help="Ollama max output tokens for Script 1 concept extraction "
                        "(default: 512)")
    g.add_argument("--num_ctx", type=int, default=4096,
                   help="Ollama context window tokens (default: 4096). "
                        "Must match your Modelfile.")
    g.add_argument("--temperature", type=float, default=0.0,
                   help="LLM temperature: 0=deterministic, 1=creative "
                        "(default: 0.0 for Script 1, 0.1 for Script 6)")
    g.add_argument("--llm_timeout", type=int, default=1200,
                   help="LLM HTTP timeout in seconds (default: 1200 = 20 min)")

    # ── Embedding ─────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Embedding")
    g.add_argument("--embed_model",
                   default="sentence-transformers/all-mpnet-base-v2",
                   help="Sentence embedding model (default: all-mpnet-base-v2)")
    g.add_argument("--sample_size", type=int, default=None,
                   help="Max sample values kept per column in Script 1 (default: 5)")

    # ── Louvain ───────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Louvain clustering")
    g.add_argument("--resolution", type=float, default=1.2,
                   help="Louvain resolution — higher = more domains (default: 1.2)")
    g.add_argument("--random_state", type=int, default=0,
                   help="Louvain random seed (default: 0)")

    # ── Control ───────────────────────────────────────────────────────────────
    g = parser.add_argument_group("Pipeline control")
    g.add_argument("--start_from", choices=STEP_ORDER, default=None,
                   help=f"Resume from a specific step. Choices: {STEP_ORDER}")
    g.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing them")
    g.add_argument("--log_dir", default="logs",
                   help="Folder for log files inside out_dir (default: logs)")
    g.add_argument("--clean", action="store_true",
                   help="Remove all previous ccm_output/ contents before running "
                        "(skipped when --start_from is used)")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()