"""
extract_schema.py
=================
Builds a schema.json from any folder of CSV files.
Works for any dataset: Northwind, Chinook, eICU, Spider, TPC, etc.

Changes vs previous version:
  - Schema is written to the output file INCREMENTALLY after each column
    (safe to inspect / resume even if the run is interrupted mid-way)
  - A progress bar shows overall percentage, current table, and ETA
  - Terminal output is minimal — only the progress bar and errors

Output format (matches CCM pipeline schema.json):
{
    "_meta": {
        "version":     "1.0",
        "description": "<database> schema -- <n> tables",
        "database":    "<database>",
        "created":     "<today>"
    },
    "tables": {
        "TableName": {
            "_description": "<LLM table-level description>",
            "columns": [
                {
                    "name":        str,
                    "sql_type":    str,
                    "key":         str,   -- "PK" / "FK" / ""
                    "description": str,   -- LLM-generated
                    "samples":     list   -- up to 5 distinct non-null values
                },
                ...
            ]
        },
        ...
    }
}

Usage:
    # Northwind
    python extract_schema.py --csv_dir csv --output northwind_schema.json
                             --database Northwind

    # eICU
    python extract_schema.py --csv_dir eicu/csv --output eicu/schema.json
                             --database eICU_Demo

    # Any Spider dataset
    python extract_schema.py --csv_dir spider/university_1/csv
                             --output spider/university_1/schema.json
                             --database university_1

    # Without LLM (fast test)
    python extract_schema.py --csv_dir csv --output schema.json --no_llm
"""

import argparse
import csv
import json
import logging
import re
import shutil
import sys
import time
from datetime import date
from pathlib import Path

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("[WARNING] 'requests' not installed. Run: pip install requests")

try:
    from pipeline_utils import setup_log_file, clean_extract_schema
    _UTILS = True
except ImportError:
    _UTILS = False

log = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_LLM_MODEL  = "mistral"

# Simple in-memory cache: (col_name, table_name) -> description
_desc_cache: dict = {}


# =============================================================================
# Progress bar
# =============================================================================

class ProgressBar:
    """
    Single-line terminal progress bar.
    Shows: percentage | filled bar | current item | elapsed | ETA
    All LLM output is suppressed — only this bar is shown.
    """

    def __init__(self, total: int, label: str = ""):
        self.total    = max(total, 1)
        self.current  = 0
        self.label    = label
        self.start_t  = time.time()
        self._width   = shutil.get_terminal_size(fallback=(80, 24)).columns
        self._msg     = ""   # current item description shown on the right
        self._draw()

    def update(self, step: int = 1, msg: str = ""):
        self.current = min(self.current + step, self.total)
        if msg:
            self._msg = msg
        self._draw()

    def set_msg(self, msg: str):
        self._msg = msg
        self._draw()

    def finish(self, final_msg: str = ""):
        self.current = self.total
        self._msg    = final_msg or "done"
        self._draw()
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _draw(self):
        elapsed = time.time() - self.start_t
        pct     = self.current / self.total

        # ETA
        if self.current > 0 and pct < 1.0:
            eta_s = elapsed / pct * (1 - pct)
            eta   = _fmt_time(eta_s)
        else:
            eta   = "00:00"

        elapsed_str = _fmt_time(elapsed)

        # Fixed parts
        prefix = f"{int(pct*100):3d}%"
        suffix = f" {elapsed_str}<{eta}  {self._msg}"

        # Bar width — fill remaining space
        bar_space = self._width - len(prefix) - len(suffix) - 4
        bar_space = max(bar_space, 4)
        filled    = int(bar_space * pct)
        bar       = "█" * filled + "░" * (bar_space - filled)

        line = f"\r{prefix} |{bar}| {suffix}"

        # Truncate to terminal width to avoid wrapping
        if len(line) > self._width:
            line = line[: self._width - 1]

        sys.stdout.write(line)
        sys.stdout.flush()


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# =============================================================================
# Incremental JSON writer
# =============================================================================

class IncrementalSchemaWriter:
    """
    Writes schema.json to disk after every column is processed.
    The file is always valid JSON — safe to open mid-run.
    """

    def __init__(self, output_path: Path, meta: dict):
        self.output_path = output_path
        self._meta       = meta
        self._tables: dict = {}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._flush()

    def add_table(self, table_name: str, description: str):
        """Register a new table (no columns yet)."""
        self._tables[table_name] = {
            "_description": description,
            "columns":      [],
        }
        self._flush()

    def add_column(self, table_name: str, col: dict):
        """Append a column to an existing table and flush to disk."""
        self._tables[table_name]["columns"].append(col)
        self._flush()

    def update_meta_description(self, description: str):
        self._meta["description"] = description
        self._flush()

    def _flush(self):
        schema = {"_meta": self._meta, "tables": self._tables}
        tmp    = self.output_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(schema, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.output_path)   # atomic rename — never corrupts file

    @property
    def tables(self):
        return self._tables


# =============================================================================
# Type inference
# =============================================================================

def infer_sql_type(values: list) -> str:
    non_null = [v for v in values if v not in ("", None, "\\N", "NULL", "nan")]
    if not non_null:
        return "TEXT"

    bool_vals = {"true", "false", "0", "1", "t", "f", "yes", "no"}
    if all(str(v).strip().lower() in bool_vals for v in non_null):
        return "BOOLEAN"

    def is_int(v):
        try:
            int(str(v).strip()); return True
        except ValueError:
            return False

    def is_float(v):
        try:
            float(str(v).strip()); return True
        except ValueError:
            return False

    if all(is_int(v) for v in non_null):
        return "INT"
    if all(is_float(v) for v in non_null):
        return "FLOAT"

    max_len = max(len(str(v)) for v in non_null)
    if max_len <= 10:   return f"VARCHAR({max(10, max_len)})"
    if max_len <= 50:   return "VARCHAR(50)"
    if max_len <= 255:  return "VARCHAR(255)"
    return "TEXT"


# =============================================================================
# Key detection
# =============================================================================

def infer_key(col_name: str, table_name: str) -> str:
    name_lower  = col_name.lower()
    table_lower = table_name.lower().replace("_", "")

    id_patterns = [
        r"^(.+?)(?:unitstayid|stayid|systemstayid|id)$",
        r"^(.+?)_id$",
    ]
    for pat in id_patterns:
        m = re.match(pat, name_lower)
        if m:
            base = m.group(1).rstrip("_").replace("_", "")
            if table_lower.startswith(base) or base in table_lower:
                return "PK"
            return "FK"
    return ""


# =============================================================================
# LLM helpers  (stderr used so they don't interfere with progress bar)
# =============================================================================

def _llm_post(ollama_url: str, model: str, prompt: str, timeout: int) -> str | None:
    """POST to Ollama and return the response text, or None on failure."""
    try:
        resp = requests.post(
            ollama_url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        if resp.status_code == 200:
            raw = resp.json().get("response", "").strip()
            raw = raw.split("\n")[0].strip().strip('"').strip("'")
            if raw and len(raw) > 10:
                return raw
    except Exception as exc:
        log.warning("[LLM ERROR] %s", exc)
    return None


def _llm_column_description(
    col_name, table_name, samples, sql_type, database, ollama_url, model,
    timeout: int = 600
) -> str | None:
    clean_samples = [str(s) for s in samples if s not in (None, "")][:5]
    # Truncate each sample to 100 chars to avoid XML/text blowup in Ollama
    clean_samples = [s[:100] for s in clean_samples]
    samples_str   = ", ".join(clean_samples) if clean_samples else "N/A"

    prompt = (
        f"You are a database documentation expert for the {database} database.\n\n"
        f"Write exactly ONE short sentence (max 15 words) describing what the "
        f"following database column stores.\n\n"
        f"Table        : {table_name}\n"
        f"Column       : {col_name}\n"
        f"SQL type     : {sql_type}\n"
        f"Sample values: {samples_str}\n\n"
        f"Rules:\n"
        f"- One sentence only, no bullet points\n"
        f"- Be specific and precise based on the sample values\n"
        f"- Do NOT start with 'This column' or 'The column'\n"
        f"- Do NOT include the column name in the description\n"
        f"- Return ONLY the description sentence, nothing else\n"
    )
    return _llm_post(ollama_url, model, prompt, timeout=timeout)


def _llm_table_description(
    table_name, col_names, database, ollama_url, model, timeout: int = 600
) -> str:
    cols_str = ", ".join(str(c) for c in col_names[:20] if c is not None)
    prompt = (
        f"You are a database documentation expert for the {database} database.\n\n"
        f"Write exactly ONE sentence (max 20 words) describing the business purpose "
        f"of the following database table.\n\n"
        f"Table  : {table_name}\n"
        f"Columns: {cols_str}\n\n"
        f"Rules:\n"
        f"- One sentence only\n"
        f"- Describe what real-world entity or process this table captures\n"
        f"- Do NOT start with 'This table' or 'The table'\n"
        f"- Return ONLY the description sentence, nothing else\n"
    )
    result = _llm_post(ollama_url, model, prompt, timeout=timeout)
    return result or f"Stores {table_name.replace('_', ' ').lower()} records."


def _llm_meta_description(
    database, table_names, ollama_url, model, timeout: int = 600
) -> str:
    tables_str = ", ".join(table_names)
    prompt = (
        f"You are a database documentation expert.\n\n"
        f"Write exactly ONE sentence (max 20 words) describing the overall purpose "
        f"of the following database.\n\n"
        f"Database name: {database}\n"
        f"Tables       : {tables_str}\n\n"
        f"Rules:\n"
        f"- One sentence only\n"
        f"- Describe what business domain or system this database supports\n"
        f"- Do NOT start with 'This database' or 'The database'\n"
        f"- Return ONLY the description sentence, nothing else\n"
    )
    result = _llm_post(ollama_url, model, prompt, timeout=timeout)
    return result or f"{database} schema -- {len(table_names)} tables"


def generate_column_description(
    col_name, table_name, key, samples, sql_type, database, ollama_url, model,
    timeout: int = 600
) -> str:
    # PK / FK — deterministic, no LLM needed
    if key == "PK":
        return f"Primary key uniquely identifying each row in {table_name}"
    if key == "FK":
        ref = re.sub(
            r"(unitstayid|stayid|systemstayid|id)$", "", col_name,
            flags=re.IGNORECASE
        )
        ref = re.sub(r"([A-Z])", r" \1", ref).strip()
        return f"Foreign key linking to the {ref.strip()} entity"

    # Cache check
    cache_key = (col_name.lower(), table_name.lower())
    if cache_key in _desc_cache:
        return _desc_cache[cache_key]

    # LLM — fall back gracefully on timeout or failure
    desc = _llm_column_description(
        col_name, table_name, samples, sql_type, database, ollama_url, model,
        timeout=timeout
    )

    if not desc:
        desc = (f"Stores {col_name.replace('_', ' ').lower()} "
                f"values for {table_name.replace('_', ' ').lower()} records.")
        log.warning("[fallback] LLM failed for %s.%s — using rule-based description.",
                    table_name, col_name)

    _desc_cache[cache_key] = desc
    return desc


# =============================================================================
# Main schema builder  — incremental write + progress bar
# =============================================================================

def build_schema(
    csv_dir:    Path,
    output_path: Path,
    database:   str,
    n_samples:  int  = 5,
    use_llm:    bool = True,
    ollama_url: str  = DEFAULT_OLLAMA_URL,
    model:      str  = DEFAULT_LLM_MODEL,
    llm_timeout: int = 600,
) -> None:

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {csv_dir}")

    # ── Pre-scan: count total columns for progress bar ────────────────────────
    log.info("Scanning %d CSV files ...", len(csv_files))
    table_data: list[tuple[Path, list]] = []   # (path, rows)
    total_columns = 0

    for csv_path in csv_files:
        try:
            rows = []
            with open(csv_path, encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            if rows:
                table_data.append((csv_path, rows))
                total_columns += len(rows[0].keys())
        except Exception as exc:
            print(f"\n[WARNING] {csv_path.name}: {exc}", file=sys.stderr)

    if not table_data:
        raise RuntimeError("All CSV files were empty or unreadable.")

    # total work units: 1 per table description + 1 per column + 1 for meta
    total_work = len(table_data) + total_columns + 1

    # ── Initialise incremental writer ─────────────────────────────────────────
    meta = {
        "version":     "1.0",
        "description": f"{database} schema -- {len(table_data)} tables",
        "database":    database,
        "created":     str(date.today()),
    }
    writer = IncrementalSchemaWriter(output_path, meta)

    # ── Progress bar ──────────────────────────────────────────────────────────
    bar = ProgressBar(total_work, label=database)

    # ── Process tables ────────────────────────────────────────────────────────
    for csv_path, rows in table_data:
        table_name = csv_path.stem
        col_names  = [k for k in rows[0].keys() if k is not None]

        # Table description
        bar.set_msg(f"{table_name} — table desc")
        if use_llm:
            table_desc = _llm_table_description(
                table_name, col_names, database, ollama_url, model,
                timeout=llm_timeout
            )
        else:
            table_desc = f"Stores {table_name.replace('_', ' ').lower()} records."

        writer.add_table(table_name, table_desc)
        bar.update(step=1, msg=f"{table_name} — table desc ✓")

        # Columns
        for col_name in col_names:
            bar.set_msg(f"{table_name}.{col_name}")

            all_values = [row.get(col_name, "") for row in rows]
            non_null   = [v for v in all_values
                          if v not in ("", None, "\\N", "NULL", "nan")]

            sql_type    = infer_sql_type(all_values)
            key         = infer_key(col_name, table_name)
            pre_samples = [v for v in non_null[:50] if v not in (None, "")]

            if use_llm:
                description_col = generate_column_description(
                    col_name, table_name, key, pre_samples, sql_type,
                    database, ollama_url, model, timeout=llm_timeout
                )
            else:
                if key == "PK":
                    description_col = (
                        f"Primary key uniquely identifying each row in {table_name}"
                    )
                elif key == "FK":
                    ref = re.sub(r"(id)$", "", col_name, flags=re.IGNORECASE)
                    description_col = f"Foreign key linking to the {ref} entity"
                else:
                    description_col = (
                        col_name.replace("_", " ").capitalize() + " value"
                    )

            # Build typed samples
            samples = []
            seen    = []
            for v in non_null:
                if v not in seen:
                    seen.append(v)
                    if sql_type == "INT":
                        try:    samples.append(int(v))
                        except: samples.append(v)
                    elif sql_type == "FLOAT":
                        try:    samples.append(float(v))
                        except: samples.append(v)
                    else:
                        samples.append(str(v))
                if len(samples) >= n_samples:
                    break
            while len(samples) < n_samples:
                samples.append(None)

            col_record = {
                "name":        col_name,
                "sql_type":    sql_type,
                "key":         key,
                "description": description_col,
                "samples":     samples,
            }

            # ── INCREMENTAL WRITE after every column ──────────────────────────
            writer.add_column(table_name, col_record)
            bar.update(step=1, msg=f"{table_name}.{col_name} ✓")

    # ── _meta description ─────────────────────────────────────────────────────
    bar.set_msg("_meta description")
    if use_llm:
        meta_desc = _llm_meta_description(
            database, list(writer.tables.keys()), ollama_url, model,
            timeout=llm_timeout
        )
        writer.update_meta_description(meta_desc)
    bar.update(step=1, msg="_meta ✓")

    bar.finish("complete")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_cols = sum(len(t["columns"]) for t in writer.tables.values())
    summary = [
        "",
        "=" * 60,
        f"Schema written : {output_path}",
        f"  Tables       : {len(writer.tables)}",
        f"  Columns      : {total_cols}",
        f"  File size    : {output_path.stat().st_size / 1024:.1f} KB",
        "=" * 60,
    ]
    for line in summary:
        print(line)
        log.info(line)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build schema.json from a folder of CSV files (any dataset).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_schema.py --csv_dir csv --output northwind_schema.json --database Northwind
  python extract_schema.py --csv_dir csv --output eicu_schema.json --database eICU_Demo
  python extract_schema.py --csv_dir csv --output schema.json --no_llm
"""
    )
    parser.add_argument("--csv_dir",    default="csv",
                        help="Folder containing CSV files (default: ./csv)")
    parser.add_argument("--output",     default="schema.json",
                        help="Output path for schema.json (default: ./schema.json)")
    parser.add_argument("--database",   default=None,
                        help="Database name (default: inferred from folder name)")
    parser.add_argument("--samples",    type=int, default=5,
                        help="Number of sample values per column (default: 5)")
    parser.add_argument("--no_llm",     action="store_true",
                        help="Skip LLM — use rule-based descriptions (fast test only)")
    parser.add_argument("--model",      default=DEFAULT_LLM_MODEL,
                        help=f"Ollama model name (default: {DEFAULT_LLM_MODEL})")
    parser.add_argument("--ollama_url", default=DEFAULT_OLLAMA_URL,
                        help="Ollama API endpoint")
    parser.add_argument("--ollama_timeout", type=int, default=600,
                        help="Ollama request timeout in seconds (default: 600). "
                             "Increase for large XML/text columns.")
    parser.add_argument("--dataset_dir", default=None,
                        help="Dataset working directory (csv/, schema.json output). "
                             "Use when extract_schema.py lives in a parent folder: "
                             "--dataset_dir Chinook. Defaults to current working directory.")
    parser.add_argument("--log_dir",    default="logs",
                        help="Folder for log files (default: ./logs)")
    parser.add_argument("--no_clean",   action="store_true",
                        help="Skip removal of schema.json before running")
    args = parser.parse_args()

    # ── Dataset directory — chdir so all relative paths resolve correctly ────
    if args.dataset_dir is not None:
        import os as _os
        _os.chdir(args.dataset_dir)


    csv_dir  = Path(args.csv_dir).resolve()
    out_path = Path(args.output).resolve()
    database = args.database or csv_dir.parent.name or csv_dir.name
    use_llm  = not args.no_llm

    # ── Logging — folder next to this script, output to console + file ────────
    script_dir = Path(__file__).resolve().parent
    log_dir    = script_dir / args.log_dir
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    if _UTILS:
        setup_log_file("extract_schema", log_dir=str(log_dir))
    else:
        from datetime import datetime
        log_dir.mkdir(parents=True, exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        _lp = log_dir / f"extract_schema_{ts}.log"
        _fh = logging.FileHandler(_lp, encoding="utf-8")
        _fh.setFormatter(logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S"))
        logging.getLogger().addHandler(_fh)
        log.info("Log file: %s", _lp)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if not args.no_clean:
        if _UTILS:
            clean_extract_schema(out_path)
        elif out_path.exists():
            out_path.unlink()
            log.info("[clean] Removed %s", out_path)

    if use_llm and not REQUESTS_AVAILABLE:
        raise RuntimeError(
            "LLM is enabled but 'requests' is not installed.\n"
            "Run: pip install requests"
        )

    # ── Banner — print and log ────────────────────────────────────────────────
    banner = [
        "=" * 60,
        "Schema Builder — CCM Pipeline",
        f"  CSV folder   : {csv_dir}",
        f"  Output       : {out_path}",
        f"  Database     : {database}",
        f"  Descriptions : {'LLM (' + args.model + ')' if use_llm else 'rule-based (--no_llm)'}",
        f"  Ollama URL   : {args.ollama_url}",
        f"  Log dir      : {log_dir}",
        "=" * 60,
        "",
    ]
    for line in banner:
        print(line)
        log.info(line)

    build_schema(
        csv_dir     = csv_dir,
        output_path = out_path,
        database    = database,
        n_samples   = args.samples,
        use_llm     = use_llm,
        ollama_url  = args.ollama_url,
        model       = args.model,
        llm_timeout = args.ollama_timeout,
    )


if __name__ == "__main__":
    main()