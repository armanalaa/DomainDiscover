"""
pipeline_utils.py
=================
Shared utilities for the CCM pipeline scripts:
  - Logging to a timestamped file in a logs/ folder
  - Cleanup of output files before a fresh run

Used by: extract_data.py, extract_schema.py,
         extract_knowledge.py, run_pipeline.py
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path


# =============================================================================
# LOGGING
# =============================================================================

def setup_log_file(script_name: str,
                   log_dir: str | Path = "logs") -> Path:
    """
    Create a timestamped log file in log_dir and attach it to the root logger.
    Console output is preserved. Returns the log file Path.
    """
    log_dir  = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{script_name}_{ts}.log"

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)
    logging.getLogger(__name__).info("Log file: %s", log_path.resolve())
    return log_path


# =============================================================================
# CLEANUP HELPERS
# =============================================================================

def _remove_items(paths: list[Path], label: str) -> int:
    removed = 0
    for p in paths:
        try:
            if p.is_file():
                p.unlink()
                logging.info("[clean] Removed file : %s", p)
                removed += 1
            elif p.is_dir():
                shutil.rmtree(p)
                logging.info("[clean] Removed dir  : %s", p)
                removed += 1
        except Exception as exc:
            logging.warning("[clean] Could not remove %s: %s", p, exc)
    if removed:
        logging.info("[clean] %s — %d item(s) removed.", label, removed)
    else:
        logging.info("[clean] %s — nothing to remove.", label)
    return removed


def clean_extract_data(csv_dir: str | Path = "csv") -> int:
    """Remove all .csv files from csv_dir before fresh data extraction."""
    csv_dir = Path(csv_dir)
    if not csv_dir.exists():
        return 0
    files = list(csv_dir.glob("*.csv"))
    return _remove_items(files, f"extract_data csv_dir={csv_dir}")


def clean_extract_schema(output: str | Path = "schema.json") -> int:
    """Remove schema.json before fresh schema build."""
    return _remove_items([Path(output)], f"extract_schema output={output}")


def clean_extract_knowledge(output:   str | Path = "knowledge.docx",
                             pdf_dir: str | Path | None = None) -> int:
    """Remove knowledge.docx, its .txt sibling, and chunks/ before fresh run."""
    targets: list[Path] = [Path(output)]
    txt = Path(output).with_suffix(".txt")
    if txt.exists():
        targets.append(txt)
    if pdf_dir:
        chunks = Path(pdf_dir) / "chunks"
        if chunks.exists():
            targets.append(chunks)
    return _remove_items(targets, f"extract_knowledge output={output}")


def make_run_tag(theta_a: float, theta_t: float, resolution: float) -> str:
    """
    Return a compact, filesystem-safe tag encoding the three tunable parameters.

    Example: theta_a=0.65, theta_t=0.70, resolution=1.2  →  "tA0.65_tT0.70_r1.2"

    Used by run_pipeline.py to name the per-run subfolder inside ccm_output/:
        ccm_output/tA0.65_tT0.70_r1.2/
    Steps 1+2, 3a, 3b write to ccm_output/ (shared, parameter-independent).
    Steps 3c, 4, 5 write to ccm_output/<run_tag>/ (parameter-specific).
    """
    return f"tA{theta_a}_tT{theta_t}_r{resolution}"


def clean_run_pipeline(out_dir: str | Path = "ccm_output",
                       run_tag: str | None = None) -> int:
    """
    Clean pipeline outputs before a fresh run.

    Two modes:
      run_tag is None  — full clean: remove everything inside out_dir
                         except the top-level logs/ subfolder and any
                         run-tagged subdirectories (tA*_tT*_r* folders).
                         Used when re-running from Step 1.
      run_tag is given — targeted clean: remove only the specific
                         ccm_output/<run_tag>/ subfolder.
                         Used when re-running from Step 3c onward with
                         the same parameters.
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return 0

    if run_tag is not None:
        # Remove only this run's subfolder
        run_dir = out_dir / run_tag
        return _remove_items([run_dir], f"run_pipeline run_dir={run_dir}")
    else:
        # Full clean — keep logs/ and any existing run subfolders
        targets = [
            p for p in out_dir.iterdir()
            if p.name != "logs" and not _is_run_dir(p)
        ]
        return _remove_items(targets, f"run_pipeline out_dir={out_dir}")


def _is_run_dir(p: Path) -> bool:
    """Return True if the path looks like a run-tagged subfolder (tA*_tT*_r*)."""
    return p.is_dir() and p.name.startswith("tA") and "_tT" in p.name and "_r" in p.name