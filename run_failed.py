"""
run_failed.py
=============
Reads tune_params_results.xlsx (or scans ccm_output/ directly),
finds all FAILED runs, retries them one by one, and rebuilds the
summary after each successful completion.

Usage:
    python run_failed.py --dataset_dir Synthea
    python run_failed.py --dataset_dir Synthea --max_retries 3
    python run_failed.py --dataset_dir Synthea --no_llm

Place in DomainMiner\ root alongside run_pipeline.py.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL      = "mistral:latest"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_SCHEMA     = "schema.json"
DEFAULT_KNOWLEDGE  = "knowledge.docx"
DEFAULT_MAX_RETRIES = 2
TIMEOUT_SECONDS    = 1800   # 30 min per run before giving up

# =============================================================================
# Helpers
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def find_failed_runs(dataset_dir: Path) -> list[dict]:
    """Scan ccm_output/ for runs missing a valid step5_report.txt with Q."""
    out_dir = dataset_dir / "ccm_output"
    if not out_dir.exists():
        log(f"ERROR: ccm_output/ not found in {dataset_dir}")
        return []

    pattern = re.compile(r"tA([\d.]+)_tT([\d.]+)_r([\d.]+)")
    failed  = []

    for folder in sorted(out_dir.iterdir()):
        if not folder.is_dir():
            continue
        m = pattern.fullmatch(folder.name)
        if not m:
            continue

        report = folder / "step5_report.txt"
        q_found = False
        if report.exists():
            text = report.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"Modularity Q\s*:\s*-?[\d.]+", text):
                q_found = True

        if not q_found:
            failed.append({
                "theta_a":    float(m.group(1)),
                "theta_t":    float(m.group(2)),
                "resolution": float(m.group(3)),
                "run_tag":    folder.name,
            })

    return failed


def run_one(dataset_dir: str, theta_a: float, theta_t: float,
            resolution: float, schema: str, knowledge: str,
            model: str, ollama_url: str, no_llm: bool) -> bool:
    """Run a single pipeline combination. Returns True if successful."""
    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_pipeline.py"),
        "--dataset_dir", dataset_dir,
        "--schema",      schema,
        "--knowledge",   knowledge,
        "--clean",
        "--theta_a",     str(theta_a),
        "--theta_t",     str(theta_t),
        "--resolution",  str(resolution),
        "--model",       model,
        "--ollama_url",  ollama_url,
    ]
    if no_llm:
        cmd.append("--no_llm")

    log(f"Running: tA={theta_a} tT={theta_t} res={resolution}")
    log(f"Command: {' '.join(cmd)}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            timeout=TIMEOUT_SECONDS,
        )
        elapsed = round(time.time() - start, 1)
        if result.returncode == 0:
            log(f"  -> OK ({elapsed}s)")
            return True
        else:
            log(f"  -> FAILED returncode={result.returncode} ({elapsed}s)")
            return False
    except subprocess.TimeoutExpired:
        elapsed = round(time.time() - start, 1)
        log(f"  -> TIMEOUT after {elapsed}s")
        return False
    except Exception as e:
        log(f"  -> ERROR: {e}")
        return False


def rebuild_summary(dataset_dir: str):
    """Call build_tune_params_results.py to refresh Excel + txt summary."""
    script = SCRIPT_DIR / "build_tune_params_results.py"
    if not script.exists():
        log("WARNING: build_tune_params_results.py not found — skipping summary rebuild.")
        return
    subprocess.run(
        [sys.executable, str(script), "--dataset_dir", dataset_dir],
        cwd=str(SCRIPT_DIR),
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Retry all FAILED tune_params runs automatically."
    )
    parser.add_argument("--dataset_dir", required=True,
                        help="Dataset folder name (e.g. Synthea)")
    parser.add_argument("--schema",      default=DEFAULT_SCHEMA)
    parser.add_argument("--knowledge",   default=DEFAULT_KNOWLEDGE)
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--ollama_url",  default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help="Max retry attempts per failed run (default: 2)")
    parser.add_argument("--no_llm",      action="store_true",
                        help="Skip LLM domain labeling (faster, Q-only)")
    args = parser.parse_args()

    dataset_path = (SCRIPT_DIR / args.dataset_dir).resolve()

    print(f"\n{'='*60}")
    log(f"run_failed.py starting")
    log(f"Dataset    : {dataset_path}")
    log(f"Max retries: {args.max_retries}")
    log(f"No LLM     : {args.no_llm}")
    print(f"{'='*60}\n")

    # ── Find failed runs ──────────────────────────────────────────────────────
    failed = find_failed_runs(dataset_path)

    if not failed:
        log("No failed runs found. All runs completed successfully.")
        rebuild_summary(args.dataset_dir)
        return

    log(f"Found {len(failed)} failed run(s) to retry:")
    for r in failed:
        log(f"  {r['run_tag']}")
    print()

    # ── Retry each failed run ─────────────────────────────────────────────────
    still_failed = []

    for i, run in enumerate(failed, 1):
        log(f"[{i}/{len(failed)}] Retrying {run['run_tag']} ...")

        success = False
        for attempt in range(1, args.max_retries + 1):
            if attempt > 1:
                log(f"  Attempt {attempt}/{args.max_retries} ...")
                time.sleep(5)  # brief pause between retries

            ok = run_one(
                dataset_dir = args.dataset_dir,
                theta_a     = run["theta_a"],
                theta_t     = run["theta_t"],
                resolution  = run["resolution"],
                schema      = args.schema,
                knowledge   = args.knowledge,
                model       = args.model,
                ollama_url  = args.ollama_url,
                no_llm      = args.no_llm,
            )
            if ok:
                success = True
                break

        if success:
            log(f"  {run['run_tag']} -> COMPLETED")
            # Rebuild summary after each success so progress is saved
            rebuild_summary(args.dataset_dir)
        else:
            log(f"  {run['run_tag']} -> STILL FAILED after {args.max_retries} attempt(s)")
            still_failed.append(run)

        print()

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"{'='*60}")
    log(f"Finished. {len(failed) - len(still_failed)}/{len(failed)} runs completed.")

    if still_failed:
        log(f"Still failed ({len(still_failed)}):")
        for r in still_failed:
            log(f"  {r['run_tag']}")
        log("Tip: try running with --no_llm to skip domain labeling and just get Q scores.")
    else:
        log("All previously failed runs completed successfully.")

    rebuild_summary(args.dataset_dir)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()