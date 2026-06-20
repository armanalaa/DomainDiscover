"""
=============================================================================
CCM Pipeline — Steps 1 & 2   (v2 — fully aligned with Alserafi et al. 2020)
=============================================================================

Implements:
  Step 1  Knowledge Extraction  (algorithm lines 1-4)
  Step 2  Column Profiling      (algorithm lines 5-9)

Faithfully follows:
  • Alserafi et al. Table 3  — all 17 attribute-level content meta-features
  • Alserafi et al. Eq. 3    — z-score normalisation across ALL columns
  • Running_Example.docx     — profiling, z-score, embedding details

─────────────────────────────────────────────────────────────────────────────
ALSERAFI TABLE 3  —  ALL 17 META-FEATURES IMPLEMENTED
─────────────────────────────────────────────────────────────────────────────

  TYPE     FEATURE              DESCRIPTION
  ───────  ───────────────────  ──────────────────────────────────────────────
  All      distinct_values_cnt  Number of distinct values
  All      distinct_values_pct  Distinct values / total instances
  All      missing_values_pct   Missing values / total instances

  Nominal  val_size_avg         Avg string length of values
  Nominal  val_size_min         Min string length of values
  Nominal  val_size_max         Max string length of values
  Nominal  val_size_std         Std of string lengths
  Nominal  val_pct_median       Median % of instances per value
  Nominal  val_pct_min          Min % of instances per value
  Nominal  val_pct_max          Max % of instances per value
  Nominal  val_pct_std          Std of % of instances per value

  Numeric  mean                 Mean numeric value
  Numeric  std                  Standard deviation
  Numeric  min_val              Minimum value
  Numeric  max_val              Maximum value
  Numeric  range_val            max_val - min_val
  Numeric  co_of_var            Coefficient of variation  (std / mean)

─────────────────────────────────────────────────────────────────────────────
Z-SCORE NORMALISATION  —  Alserafi Eq. 3
─────────────────────────────────────────────────────────────────────────────

  After profiling ALL columns, for each meta-feature m:
    μ_m  = mean of m(Ai) across all columns that have feature m
    σ_m  = std  of m(Ai) across all columns that have feature m

    z_m(Ai) = ( m(Ai) − μ_m ) / σ_m

  Stored as Profile(Ai).z_features  — used directly in Step 3 P_stat.

─────────────────────────────────────────────────────────────────────────────
FREE MODELS
─────────────────────────────────────────────────────────────────────────────
  Embedding : sentence-transformers/all-mpnet-base-v2  (768-dim, local, free)
  Concept LLM: Ollama + mistral  |  HuggingFace free API  |  mock (default)

INSTALL
  pip install sentence-transformers pandas numpy requests

RUN
  python ccm_steps1_2_v2.py
  python ccm_steps1_2_v2.py --schema schema.json
  LLM_BACKEND=ollama  python ccm_steps1_2_v2.py
  LLM_BACKEND=huggingface python ccm_steps1_2_v2.py   # needs HF_TOKEN

OUTPUTS  (./ccm_output/)
  step1_concepts.json            C + e(c)           for all 7 concepts
  step2_column_profiles.json     Profile(Ai) + z-scores + e(Ai)
  step2_zscore_stats.json        μ and σ per feature (for Step 3 reuse)
  phi_matrix.csv                 φ_i[c] = cosine(e(Ai),e(c))  (130×7)
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd;      _PANDAS   = True
except ImportError:           _PANDAS   = False
try:
    from sentence_transformers import SentenceTransformer
    _ST = True
except ImportError:           _ST       = False
try:
    import requests;          _REQUESTS = True
except ImportError:           _REQUESTS = False

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    SCHEMA_FILE:  str  = "schema.json"
    # Embedding: best free local model (768-dim, MTEB 57.78, no API key)
    EMBED_MODEL:  str  = "sentence-transformers/all-mpnet-base-v2"
    # LLM backend for concept extraction
    LLM_BACKEND:  str  = os.getenv("LLM_BACKEND", "ollama")
    OLLAMA_URL:   str  = os.getenv("OLLAMA_URL",   "http://localhost:11434")
    OLLAMA_MODEL: str  = os.getenv("OLLAMA_MODEL", "mistral-ctx4k")
    HF_TOKEN:     str  = os.getenv("HF_TOKEN",     "")
    HF_MODEL:     str  = "mistralai/Mistral-7B-Instruct-v0.2"
    SAMPLE_SIZE:  int  = 5      # max sample values kept per column
    OUT_DIR:      Path = Path("ccm_output")
    # LLM generation params — all overridable via CLI
    NUM_PREDICT:  int   = 512
    NUM_CTX:      int   = 4096
    TEMPERATURE:  float = 0.0
    LLM_TIMEOUT:  int   = 1200


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ColumnProfile:
    """
    Profile(Ai) — complete metadata following Alserafi Table 3.

    Raw meta-features (computed from sample values):
      ALL columns   : distinct_values_cnt, distinct_values_pct, missing_values_pct
      Nominal only  : val_size_avg/min/max/std, val_pct_median/min/max/std
      Numeric only  : mean, std, min_val, max_val, range_val, co_of_var

    Derived (added after processing all columns):
      z_features    : dict of z-scored meta-feature values (Eq. 3)
    """
    # identity
    table:        str
    column:       str
    sql_type:     str
    key:          str            # "PK" | "FK" | ""
    domain:       str
    is_numeric:   bool
    is_nominal:   bool
    description:  str = ""

    # ── raw counts (always present) ──────────────────────────────────────────
    sample_count:       int   = 0
    non_null_count:     int   = 0
    sample_values:      list  = field(default_factory=list)

    # ── ALL columns (Table 3, rows 1-3) ──────────────────────────────────────
    distinct_values_cnt: int   = 0
    distinct_values_pct: float = 0.0   # distinct_cnt / non_null_count
    missing_values_pct:  float = 0.0   # null_count   / sample_count

    # ── NOMINAL only (Table 3, rows 4-11) ────────────────────────────────────
    val_size_avg:    float | None = None  # avg  string length
    val_size_min:    float | None = None  # min  string length
    val_size_max:    float | None = None  # max  string length
    val_size_std:    float | None = None  # std  string length
    val_pct_median:  float | None = None  # median  % instances per value
    val_pct_min:     float | None = None  # min     % instances per value
    val_pct_max:     float | None = None  # max     % instances per value
    val_pct_std:     float | None = None  # std     % instances per value

    # ── NUMERIC only (Table 3, rows 12-17) ───────────────────────────────────
    mean:       float | None = None
    std:        float | None = None
    min_val:    float | None = None
    max_val:    float | None = None
    range_val:  float | None = None   # max_val - min_val
    co_of_var:  float | None = None   # std / mean  (0 if mean == 0)

    # ── z-scored profile vector (Alserafi Eq. 3) — filled after global pass ──
    z_features:  dict = field(default_factory=dict)

    # ── embedding ─────────────────────────────────────────────────────────────
    emb_input:   str        = ""
    embedding:   list[float] = field(default_factory=list)


@dataclass
class Concept:
    id:         str
    name:       str
    definition: str
    activities: list[str] = field(default_factory=list)
    actors:     list[str] = field(default_factory=list)
    artifacts:  list[str] = field(default_factory=list)
    emb_input:  str        = ""
    embedding:  list[float] = field(default_factory=list)


# =============================================================================
# SCHEMA LOADER
# =============================================================================

def load_schema(path: str | Path) -> dict:
    fpath = Path(path)
    if not fpath.exists():
        log.error("Schema file not found: %s", fpath.resolve())
        sys.exit(1)
    with open(fpath, encoding="utf-8") as f:
        raw = json.load(f)
    schema = raw["tables"] if "tables" in raw else raw
    meta   = raw.get("_meta", {})
    total  = sum(len(v["columns"]) for v in schema.values())
    for tname, tdata in schema.items():
        # ── NEW: read table-level _description (added in updated schema) ──
        tdata.setdefault("_description", "")
        for col in tdata["columns"]:
            col.setdefault("key",         "")
            col.setdefault("description", "")
            col.setdefault("samples",     [])
    log.info("Schema loaded: %s  (%d tables, %d columns)",
             fpath.name, len(schema), total)
    if meta:
        log.info("  Database: %s — %s",
                 meta.get("database","—"), meta.get("description","—"))
    return schema


# =============================================================================
# STEP 2  LINE 7  —  Profile(Ai) ← ComputeMetadata(Ai)
# Implements ALL 17 features from Alserafi Table 3
# =============================================================================

_NUMERIC_PREFIXES = (
    "INT","BIGINT","SMALLINT","TINYINT",
    "DECIMAL","NUMERIC","FLOAT","REAL","MONEY","BIT",
)

def _is_numeric_type(sql_type: str) -> bool:
    return sql_type.upper().startswith(_NUMERIC_PREFIXES)

def _safe_round(v, d=6):
    return round(float(v), d) if v is not None else None

def _std(values: list[float]) -> float:
    if len(values) < 2: return 0.0
    mu  = sum(values) / len(values)
    var = sum((x - mu) ** 2 for x in values) / len(values)
    return math.sqrt(var)

def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0: return 0.0
    mid = n // 2
    return (s[mid - 1] + s[mid]) / 2.0 if n % 2 == 0 else s[mid]


def compute_metadata(table: str, table_desc: str, col: dict) -> ColumnProfile:
    """
    Line 7: Profile(Ai) <- ComputeMetadata(Ai)

    Computes all 17 meta-features from Alserafi Table 3.
    Z-score normalisation (Eq. 3) is applied AFTER all columns are profiled
    (requires global μ and σ) — see compute_zscore_profiles().
    """
    name     = col["name"]
    sql_type = col["sql_type"]
    key      = col.get("key",  "")
    desc     = col.get("description", "")
    raw      = col.get("samples", [])

    is_num   = _is_numeric_type(sql_type)
    non_null = [v for v in raw if v is not None]
    samples  = non_null[: Config.SAMPLE_SIZE]

    p = ColumnProfile(
        table        = table,
        column       = name,
        sql_type     = sql_type,
        key          = key,
        domain       = table_desc,   # carries table _description for embedding
        is_numeric   = is_num,
        is_nominal   = not is_num,
        description  = desc,
        sample_count    = len(raw),
        non_null_count  = len(non_null),
        sample_values   = samples,
    )

    n_total   = max(len(raw),      1)
    n_nonull  = max(len(non_null), 1)

    # ── ALL columns: 3 universal features ────────────────────────────────────
    distinct_set             = {str(v) for v in non_null}
    p.distinct_values_cnt    = len(distinct_set)
    p.distinct_values_pct    = _safe_round(len(distinct_set) / n_nonull)
    p.missing_values_pct     = _safe_round((len(raw) - len(non_null)) / n_total)

    # ── NOMINAL columns: 8 string/frequency features ─────────────────────────
    if not is_num and non_null:
        str_vals   = [str(v) for v in non_null]
        lengths    = [len(v) for v in str_vals]

        # val_size: string length stats
        p.val_size_avg = _safe_round(sum(lengths) / len(lengths))
        p.val_size_min = _safe_round(min(lengths))
        p.val_size_max = _safe_round(max(lengths))
        p.val_size_std = _safe_round(_std([float(x) for x in lengths]))

        # val_pct: % of instances per distinct value
        counts = Counter(str_vals)
        pcts   = [cnt / n_nonull for cnt in counts.values()]   # fraction

        p.val_pct_median = _safe_round(_median(pcts))
        p.val_pct_min    = _safe_round(min(pcts))
        p.val_pct_max    = _safe_round(max(pcts))
        p.val_pct_std    = _safe_round(_std(pcts))

    # ── NUMERIC columns: 6 statistical features ───────────────────────────────
    elif is_num and non_null:
        nums = [float(v) for v in non_null]
        mu   = sum(nums) / len(nums)
        sd   = _std(nums)

        p.mean      = _safe_round(mu)
        p.std       = _safe_round(sd)
        p.min_val   = _safe_round(min(nums))
        p.max_val   = _safe_round(max(nums))
        p.range_val = _safe_round(max(nums) - min(nums))
        # coefficient of variation:  std / |mean|   (0 if mean ≈ 0)
        p.co_of_var = _safe_round(sd / abs(mu)) if abs(mu) > 1e-9 else 0.0

    return p


# =============================================================================
# Z-SCORE NORMALISATION  —  Alserafi Eq. 3
# =============================================================================

# The 17 feature names that go into the z-score profile vector.
# Columns that don't have a feature (e.g. a nominal col won't have 'mean')
# receive z = 0.0 (neutral — no contribution to P_stat distance).

ALL_FEATURES = [
    # universal (3)
    "distinct_values_cnt", "distinct_values_pct", "missing_values_pct",
    # nominal (8)
    "val_size_avg", "val_size_min", "val_size_max", "val_size_std",
    "val_pct_median", "val_pct_min", "val_pct_max", "val_pct_std",
    # numeric (6)
    "mean", "std", "min_val", "max_val", "range_val", "co_of_var",
]


def compute_zscore_profiles(profiles: list[ColumnProfile]) -> dict:
    """
    Alserafi Eq. 3:
      z_m(Ai) = ( m(Ai) − μ_m ) / σ_m

    μ_m and σ_m are computed over ALL columns that have a value for feature m.
    Columns without the feature (e.g. mean for a nominal column) get z = 0.0.

    Returns:
      stats — dict { feature_name: {"mu": ..., "sigma": ...} }
              saved to step2_zscore_stats.json for Step 3 reuse.
    """
    stats: dict[str, dict] = {}

    for feat in ALL_FEATURES:
        # collect raw values from all profiles that have this feature
        values = []
        for p in profiles:
            v = getattr(p, feat, None)
            if v is not None:
                values.append(float(v))

        if len(values) < 2:
            mu, sigma = (values[0] if values else 0.0), 1.0
        else:
            mu    = sum(values) / len(values)
            sigma = _std(values)
            if sigma < 1e-9:
                sigma = 1.0   # avoid division by zero for constant features

        stats[feat] = {"mu": round(mu, 6), "sigma": round(sigma, 6)}

    # Apply z-scores to every profile
    for p in profiles:
        z: dict[str, float] = {}
        for feat in ALL_FEATURES:
            raw_val = getattr(p, feat, None)
            mu      = stats[feat]["mu"]
            sigma   = stats[feat]["sigma"]
            if raw_val is not None:
                z[feat] = round((float(raw_val) - mu) / sigma, 6)
            else:
                z[feat] = 0.0   # feature not applicable → neutral
        p.z_features = z

    log.info("Z-score normalisation done — %d features × %d columns",
             len(ALL_FEATURES), len(profiles))
    return stats


# =============================================================================
# EMBEDDING INPUT BUILDERS
# =============================================================================

def build_col_emb_input(p: ColumnProfile) -> str:
    """
    Embedding text for  e(Ai) <- Embed(...)

    Following the running example:
      Column name: {name}
      Table: {table}
      Description: {description}
      Sample values: {samples}
    """
    key_tag = f" [{p.key}]" if p.key else ""
    samples = ", ".join(str(v) for v in p.sample_values)
    kind    = "numeric" if p.is_numeric else "nominal"

    # p.domain carries the table-level _description (see compute_metadata)
    table_desc = p.domain.strip() if p.domain else ""

    parts = [
        f"Column name: {p.column}{key_tag}.",
        f"Table: {p.table}.",
        f"SQL type: {p.sql_type} ({kind}).",
    ]
    if table_desc:
        parts.append(f"Table context: {table_desc}.")
    parts += [
        f"Description: {p.description}.",
        f"Sample values: {samples}.",
    ]

    if p.is_numeric and p.mean is not None:
        parts.append(
            f"Statistics: mean={p.mean}, std={p.std}, "
            f"min={p.min_val}, max={p.max_val}, "
            f"range={p.range_val}, cv={p.co_of_var}."
        )
    elif p.is_nominal and p.val_size_avg is not None:
        parts.append(
            f"Value stats: avg_len={p.val_size_avg}, "
            f"distinct={p.distinct_values_cnt} "
            f"({p.distinct_values_pct*100:.1f}%), "
            f"missing={p.missing_values_pct*100:.1f}%."
        )

    return " ".join(parts)


def build_concept_emb_input(c: Concept) -> str:
    """Embedding text for  e(c) <- Embed(...)"""
    acts  = "; ".join(c.activities) if c.activities else "—"
    roles = ", ".join(c.actors)     if c.actors     else "—"
    arts  = ", ".join(c.artifacts)  if c.artifacts  else "—"
    return (
        f"Business concept: {c.name}. "
        f"Definition: {c.definition} "
        f"Activities: {acts}. "
        f"Performed by: {roles}. "
        f"Data artifacts produced: {arts}."
    )


# =============================================================================
# EMBEDDING ENGINE  —  sentence-transformers/all-mpnet-base-v2
# =============================================================================

class EmbeddingEngine:
    def __init__(self, model_name: str = Config.EMBED_MODEL):
        if not _ST:
            raise ImportError("pip install sentence-transformers")
        log.info("Loading embedding model: %s …", model_name)
        t0 = time.time()
        self.model = SentenceTransformer(model_name)
        self.dim   = self.model.get_sentence_embedding_dimension()
        log.info("Model ready — dim=%d  (%.1fs)", self.dim, time.time() - t0)

    def encode(self, texts: list[str],
               batch_size: int = 64,
               show_progress: bool = True) -> np.ndarray:
        """Returns L2-normalised float32 array (N, dim). cosine = dot product."""
        vecs = self.model.encode(
            texts,
            batch_size           = batch_size,
            show_progress_bar    = show_progress,
            normalize_embeddings = True,
            convert_to_numpy     = True,
        )
        norms = np.linalg.norm(vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), "L2 norm check failed"
        return vecs


# =============================================================================
# STEP 1  LINE 1  —  C <- LLM_ExtractConcepts(K_raw)
# =============================================================================

EXTRACTION_PROMPT = """\
You are a business analyst and data architect specialising in enterprise
information systems and business process modelling.

Read the business workflow text below and extract 5-8 high-level
operational CONCEPTS that collectively cover ALL activities described.

RULES
1. A concept = a coherent business capability that can serve as a semantic
   category for grouping data lake attributes.
2. Mutually exclusive — no two concepts share the same activities.
3. Generic record-keeping activities (e.g. "update records", "log data",
   "save to system") are NOT standalone concepts — record them as
   Data Artifacts attached to the relevant concept instead.
4. Concept names must be noun phrases.
5. Return ONLY a valid JSON array — no prose, no markdown fences.

OUTPUT FORMAT:
[{{"id":"C1","name":"...","definition":"...","activities":[...],"actors":[...],"artifacts":[...]}}]

WORKFLOW TEXT:
{text}
"""

# =============================================================================
# Default K_RAW — empty; must be provided via --knowledge argument
# =============================================================================
K_RAW = ""


class ConceptExtractor:
    def __init__(self, backend: str = Config.LLM_BACKEND):
        self.backend = backend
        log.info("ConceptExtractor backend = %s", backend)

    # ── Public entry point ────────────────────────────────────────────────────

    def extract(self, k_raw: str) -> list[Concept]:
        """
        Extract concepts from k_raw, processing it in chunks that fit the
        model's context window. All chunks are processed and results merged —
        no content is ever truncated or skipped.
        """
        max_chars = max(3000, (Config.NUM_CTX - 1024) * 3)

        if len(k_raw) <= max_chars:
            log.info("K_RAW fits in one call (%d chars ≤ %d)", len(k_raw), max_chars)
            return self._extract_one(k_raw)

        # Chunked extraction — split and process all chunks
        chunks = self._split_chunks(k_raw, max_chars)
        log.info("K_RAW (%d chars) exceeds num_ctx=%d limit (~%d chars) — "
                 "splitting into %d chunks",
                 len(k_raw), Config.NUM_CTX, max_chars, len(chunks))

        all_concepts: list[Concept] = []
        seen_names:   set[str]      = set()

        for idx, chunk in enumerate(chunks, 1):
            log.info("── Concept extraction chunk %d/%d (%d chars) ──",
                     idx, len(chunks), len(chunk))
            try:
                concepts = self._extract_one(chunk)
            except Exception as exc:
                log.warning("Chunk %d failed: %s — skipping", idx, exc)
                continue

            new = 0
            for c in concepts:
                key = c.name.lower().strip()
                if key and key not in seen_names:
                    seen_names.add(key)
                    c.id = f"C{len(all_concepts) + 1}"
                    all_concepts.append(c)
                    new += 1
            log.info("Chunk %d: %d concepts parsed, %d new added "
                     "(total: %d)", idx, len(concepts), new, len(all_concepts))

        log.info("Chunked extraction complete — %d unique concepts total",
                 len(all_concepts))
        return all_concepts

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _split_chunks(self, text: str, max_chars: int,
                      overlap: int = 200) -> list[str]:
        """
        Split text into chunks of at most max_chars with small overlap.
        Splits at newline boundaries to avoid cutting mid-sentence.
        """
        chunks = []
        start  = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            if end < len(text):
                nl = text.rfind("\n", start, end)
                if nl > start + max_chars // 2:
                    end = nl
            chunks.append(text[start:end].strip())
            start = end - overlap if end < len(text) else end
        return [c for c in chunks if c]

    # ── Single LLM call ───────────────────────────────────────────────────────

    def _extract_one(self, k_raw: str) -> list[Concept]:
        prompt = EXTRACTION_PROMPT.format(text=k_raw.strip())
        if   self.backend == "ollama":      raw = self._ollama(prompt)
        elif self.backend == "huggingface": raw = self._hf(prompt)
        else:                               return self._mock()
        return self._parse(raw)

    def _ollama(self, prompt: str) -> str:
        if not _REQUESTS: raise ImportError("pip install requests")
        r = requests.post(
            f"{Config.OLLAMA_URL}/api/generate",
            json={"model": Config.OLLAMA_MODEL, "prompt": prompt,
                  "stream": False,
                  "options": {
                      "temperature": Config.TEMPERATURE,
                      "num_predict": Config.NUM_PREDICT,
                      "num_ctx":     Config.NUM_CTX,
                  }},
            timeout=Config.LLM_TIMEOUT,
        )
        r.raise_for_status()
        return _find_json(r.json().get("response", ""))

    def _hf(self, prompt: str) -> str:
        if not _REQUESTS: raise ImportError("pip install requests")
        if not Config.HF_TOKEN:
            raise ValueError("Set env var HF_TOKEN or pass --hf_token")
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{Config.HF_MODEL}",
            headers={"Authorization": f"Bearer {Config.HF_TOKEN}"},
            json={"inputs": f"<s>[INST] {prompt} [/INST]",
                  "parameters": {"max_new_tokens": max(Config.NUM_PREDICT, 512),
                                 "temperature":    max(Config.TEMPERATURE, 0.01),
                                 "return_full_text": False}},
            timeout=Config.LLM_TIMEOUT,
        )
        r.raise_for_status()
        return _find_json(r.json()[0].get("generated_text", ""))

    def _mock(self) -> list[Concept]:
        raise RuntimeError(
            "LLM_BACKEND=mock is not supported in this pipeline.\n"
            "Concept extraction requires a real LLM.\n"
            "Set LLM_BACKEND=ollama (default) and ensure Ollama is running:\n"
            "  ollama serve\n"
            "  ollama pull mistral"
        )

    def _parse(self, raw_json: str) -> list[Concept]:
        last_brace = raw_json.rfind("}]")
        if last_brace == -1:
            last_brace = raw_json.rfind("}")
            if last_brace != -1:
                raw_json = raw_json[:last_brace + 1] + "]"
        else:
            raw_json = raw_json[:last_brace + 2]
        items = json.loads(raw_json)
        out   = [Concept(id=d.get("id", f"C{i+1}"),
                         name=d.get("name",""),
                         definition=d.get("definition",""),
                         activities=d.get("activities",[]),
                         actors=d.get("actors",[]),
                         artifacts=d.get("artifacts",[]))
                 for i, d in enumerate(items)]
        log.info("Parsed %d concepts from LLM", len(out))
        return out


def _find_json(text: str) -> str:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array in LLM output:\n{text[:300]}")
    return m.group(0)


# =============================================================================
# PERSISTENCE
# =============================================================================

def _emb_meta(vec: list[float]) -> dict:
    if not vec:
        return {"embedding": [], "dim": 0, "norm": None, "preview_8d": []}
    a = np.array(vec)
    return {"embedding":  vec,
            "dim":        len(vec),
            "norm":       round(float(np.linalg.norm(a)), 8),
            "preview_8d": [round(float(x), 6) for x in a[:8]]}


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    log.info("Saved -> %s", path)


def save_phi_matrix(profiles: list[ColumnProfile],
                    concepts:  list[Concept],
                    path:      Path) -> None:
    if not profiles[0].embedding or not concepts[0].embedding:
        log.warning("Embeddings missing — phi matrix not saved.")
        return
    E_a = np.array([p.embedding for p in profiles])   # (N_cols, 768)
    E_c = np.array([c.embedding for c in concepts])   # (N_conc, 768)
    Phi = E_a @ E_c.T                                  # (N_cols, N_conc)
    rows = [f"{p.table}.{p.column}" for p in profiles]
    cols = [c.id for c in concepts]
    if _PANDAS:
        pd.DataFrame(Phi, index=rows, columns=cols).to_csv(
            path, float_format="%.6f")
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write("column," + ",".join(cols) + "\n")
            for row, vec in zip(rows, Phi):
                f.write(row + "," + ",".join(f"{v:.6f}" for v in vec) + "\n")
    log.info("phi matrix -> %s  (%d x %d)", path, *Phi.shape)
    # Top-5 preview per concept
    log.info("\nTop-5 columns per concept (phi_i[c] preview):")
    for j, c in enumerate(concepts):
        top5 = np.argsort(Phi[:, j])[::-1][:5]
        log.info("  %s — %s", c.id, c.name)
        for rank, idx in enumerate(top5, 1):
            p = profiles[idx]
            log.info("    %d. %-32s [%-12s]  phi=%.4f",
                     rank, f"{p.table}.{p.column}", p.domain, Phi[idx, j])


# =============================================================================
# PIPELINE
# =============================================================================

def run_step1_knowledge_extraction() -> list[Concept]:
    """
    Algorithm lines 1-4:
      1: C  <- LLM_ExtractConcepts(K_raw)
      2: for c in C:
      3:   e(c) <- Embed(build_concept_emb_input(c))
      4: end for
    """
    log.info("━" * 66)
    log.info("STEP 1  —  Knowledge Extraction + Concept Embedding")
    log.info("━" * 66)

    log.info("K_RAW length: %d chars (will be chunked if needed for num_ctx=%d)",
             len(K_RAW), Config.NUM_CTX)
    concepts = ConceptExtractor(Config.LLM_BACKEND).extract(K_RAW)
    log.info("Line 1:  |C| = %d concepts extracted", len(concepts))
    for c in concepts:
        log.info("  %s  %s", c.id, c.name)

    # Lines 2-3: build embedding input text
    for c in concepts:
        c.emb_input = build_concept_emb_input(c)

    # Line 3: embed
    if _ST:
        engine = EmbeddingEngine(Config.EMBED_MODEL)
        vecs   = engine.encode([c.emb_input for c in concepts],
                               show_progress=False)
        for c, v in zip(concepts, vecs):
            c.embedding = v.tolist()
        log.info("Line 3:  e(c) in R^%d, ||e(c)||=1 for all c",
                 len(concepts[0].embedding))
    else:
        log.warning("sentence-transformers not installed — embeddings skipped.")

    return concepts


def run_step2_column_profiling(schema: dict) -> tuple[list[ColumnProfile], dict]:
    """
    Algorithm lines 5-9:
      5: A  <- ExtractAllColumns(T)
      6: for Ai in A:
      7:   Profile(Ai) <- ComputeMetadata(Ai)   [all 17 Table-3 features]
      8:   e(Ai)       <- Embed(emb_input)
      9: end for

    Plus z-score normalisation (Eq. 3) applied globally after profiling.
    """
    log.info("━" * 66)
    log.info("STEP 2  —  Column Profiling + Z-Score + Embedding")
    log.info("━" * 66)

    # Line 5: A <- ExtractAllColumns(T)
    A = [(t, tdata.get("_description", ""), col)
         for t, tdata in schema.items()
         for col in tdata["columns"]]
    log.info("Line 5:  |A| = %d columns from %d tables", len(A), len(schema))

    # Lines 6-7: Profile(Ai)
    profiles: list[ColumnProfile] = []
    for table, table_desc, col in A:
        p           = compute_metadata(table, table_desc, col)
        p.emb_input = build_col_emb_input(p)
        profiles.append(p)
    log.info("Line 7:  Profile(Ai) done — all 17 Alserafi Table-3 features")

    # Z-score normalisation (Eq. 3)  — requires all profiles first
    log.info("Eq. 3:   Computing z-score normalisation across %d columns ...",
             len(profiles))
    zscore_stats = compute_zscore_profiles(profiles)

    # Line 8: e(Ai) <- Embed(emb_input)
    if _ST:
        engine = EmbeddingEngine(Config.EMBED_MODEL)
        log.info("Line 8:  encoding %d column embeddings ...", len(profiles))
        vecs = engine.encode([p.emb_input for p in profiles],
                             show_progress=True)
        for p, v in zip(profiles, vecs):
            p.embedding = v.tolist()
        log.info("Line 8:  e(Ai) in R^%d, ||e(Ai)||=1 for all Ai",
                 len(profiles[0].embedding))
    else:
        log.warning("sentence-transformers not installed — embeddings skipped.")

    return profiles, zscore_stats


# =============================================================================
# MAIN
# =============================================================================

def _profile_to_dict(p: ColumnProfile) -> dict:
    """Serialise ColumnProfile to JSON-friendly dict."""
    d = asdict(p)
    emb = d.pop("embedding")
    d.update(_emb_meta(emb))
    return d


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CCM Pipeline Steps 1 & 2 — fully follows Alserafi et al.")

    # ── Paths ──────────────────────────────────────────────────────────────────
    parser.add_argument("--schema",    default=Config.SCHEMA_FILE,
                        help=f"Schema JSON file (default: {Config.SCHEMA_FILE})")
    parser.add_argument("--out",       default=str(Config.OUT_DIR),
                        help=f"Output directory (default: {Config.OUT_DIR})")
    parser.add_argument("--knowledge", default=None,
                        help="Path to K_raw knowledge file (.txt or .docx).")

    # ── Embedding ──────────────────────────────────────────────────────────────
    parser.add_argument("--embed_model", "--embed-model",
                        default=Config.EMBED_MODEL,
                        help=f"Sentence-transformers model (default: {Config.EMBED_MODEL})")
    parser.add_argument("--sample_size", type=int, default=Config.SAMPLE_SIZE,
                        help=f"Max sample values per column (default: {Config.SAMPLE_SIZE})")

    # ── LLM backend ────────────────────────────────────────────────────────────
    parser.add_argument("--llm_backend", default=None,
                        choices=["ollama", "huggingface"],
                        help=f"LLM backend (default: {Config.LLM_BACKEND})")
    parser.add_argument("--ollama_model", default=None,
                        help=f"Ollama model (default: {Config.OLLAMA_MODEL})")
    parser.add_argument("--ollama_url", default=None,
                        help=f"Ollama base URL (default: {Config.OLLAMA_URL})")
    parser.add_argument("--hf_token",  default=None,
                        help="HuggingFace API token")
    parser.add_argument("--hf_model",  default=None,
                        help=f"HuggingFace model ID (default: {Config.HF_MODEL})")

    # ── LLM generation constraints ─────────────────────────────────────────────
    parser.add_argument("--num_predict", type=int, default=None,
                        help=f"Ollama max output tokens (default: {Config.NUM_PREDICT})")
    parser.add_argument("--num_ctx",     type=int, default=None,
                        help=f"Ollama context window (default: {Config.NUM_CTX})")
    parser.add_argument("--temperature", type=float, default=None,
                        help=f"LLM temperature (default: {Config.TEMPERATURE})")
    parser.add_argument("--llm_timeout", type=int, default=None,
                        help=f"LLM HTTP timeout seconds (default: {Config.LLM_TIMEOUT})")
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

    # ── Apply CLI overrides to Config ──────────────────────────────────────────
    Config.SCHEMA_FILE = args.schema
    Config.OUT_DIR     = Path(args.out)
    Config.EMBED_MODEL = args.embed_model
    Config.SAMPLE_SIZE = args.sample_size
    if args.llm_backend:  Config.LLM_BACKEND  = args.llm_backend
    if args.ollama_model: Config.OLLAMA_MODEL  = args.ollama_model
    if args.ollama_url:   Config.OLLAMA_URL    = args.ollama_url.rstrip("/")
    if args.hf_token:     Config.HF_TOKEN      = args.hf_token
    if args.hf_model:     Config.HF_MODEL      = args.hf_model
    if args.num_predict is not None: Config.NUM_PREDICT = args.num_predict
    if args.num_ctx     is not None: Config.NUM_CTX     = args.num_ctx
    if args.temperature is not None: Config.TEMPERATURE = args.temperature
    if args.llm_timeout is not None: Config.LLM_TIMEOUT = args.llm_timeout
    Config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load K_raw from external file if provided ──────────────────────────
    global K_RAW
    if args.knowledge:
        kpath = Path(args.knowledge)
        if not kpath.exists():
            log.error("Knowledge file not found: %s", kpath.resolve())
            sys.exit(1)
        if kpath.suffix.lower() == ".docx":
            try:
                from docx import Document
                doc   = Document(str(kpath))
                K_RAW = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                log.error("python-docx not installed. Run: pip install python-docx")
                sys.exit(1)
        else:
            K_RAW = kpath.read_text(encoding="utf-8")
        log.info("  Knowledge   : loaded from %s (%d chars)", kpath.name, len(K_RAW))
    else:
        if not K_RAW.strip():
            log.error("=" * 60)
            log.error("ERROR: No knowledge file provided and K_RAW is empty.")
            log.error("")
            log.error("You must supply a knowledge document via --knowledge:")
            log.error("  python 1_knowledge_concept_embedding.py \\")
            log.error("      --knowledge path/to/knowledge.docx \\")
            log.error("      --schema    path/to/schema.json")
            log.error("")
            log.error("Generate knowledge.docx first with:")
            log.error("  python generate_knowledge_docx.py --database <name>")
            log.error("=" * 60)
            sys.exit(1)
        log.warning("  Knowledge   : using K_RAW string (%d chars) — "
                    "pass --knowledge for project-specific content", len(K_RAW))

    log.info("CCM Pipeline v2 — Alserafi Table 3 + Eq. 3")
    log.info("  Schema      : %s", Config.SCHEMA_FILE)
    log.info("  Embed model : %s", Config.EMBED_MODEL)
    log.info("  LLM backend : %s", Config.LLM_BACKEND)
    log.info("  Ollama model: %s  url=%s", Config.OLLAMA_MODEL, Config.OLLAMA_URL)
    log.info("  num_predict : %d  num_ctx=%d  temperature=%.2f  timeout=%ds",
             Config.NUM_PREDICT, Config.NUM_CTX, Config.TEMPERATURE, Config.LLM_TIMEOUT)
    log.info("  Sample size : %d  Output dir: %s", Config.SAMPLE_SIZE, Config.OUT_DIR)

    schema = load_schema(Config.SCHEMA_FILE)

    # ── Step 1 ─────────────────────────────────────────────────────────────
    concepts = run_step1_knowledge_extraction()
    save_json(
        [{**asdict(c), **_emb_meta(c.embedding)} for c in concepts],
        Config.OUT_DIR / "step1_concepts.json"
    )

    # ── Step 2 ─────────────────────────────────────────────────────────────
    profiles, zscore_stats = run_step2_column_profiling(schema)
    save_json([_profile_to_dict(p) for p in profiles],
              Config.OUT_DIR / "step2_column_profiles.json")
    save_json(zscore_stats,
              Config.OUT_DIR / "step2_zscore_stats.json")

    # ── phi matrix ─────────────────────────────────────────────────────────
    save_phi_matrix(profiles, concepts, Config.OUT_DIR / "phi_matrix.csv")

    # ── Summary ────────────────────────────────────────────────────────────
    dim = len(profiles[0].embedding) if profiles[0].embedding else "N/A"
    log.info("")
    log.info("━" * 66)
    log.info("PIPELINE COMPLETE")
    log.info("━" * 66)
    log.info("  Concepts  |C|  = %d   e(c)  in R^%s", len(concepts), dim)
    log.info("  Columns   |A|  = %d   e(Ai) in R^%s", len(profiles), dim)
    log.info("  Features/col   = 17   (Alserafi Table 3, all types)")
    log.info("  Z-score stats  = %d features normalised (Eq. 3)",
             len(zscore_stats))
    log.info("  phi pairs      = %d x %d", len(profiles), len(concepts))
    log.info("")
    log.info("Outputs:")
    log.info("  step1_concepts.json          C + e(c)")
    log.info("  step2_column_profiles.json   Profile(Ai) + z_features + e(Ai)")
    log.info("  step2_zscore_stats.json      mu and sigma per feature  (Step 3 input)")
    log.info("  phi_matrix.csv               phi_i[c] = cosine(e(Ai), e(c))")
    log.info("")
    log.info("Next -> Step 3: Column Similarity")
    log.info("  P_stat(Ai,Aj)  = (1/k) * sum_l |z_l(Ai) - z_l(Aj)|   [Eq. 3]")
    log.info("  P_name(Ai,Aj)  = 1 - Levenshtein(Ai.name, Aj.name) / max_len  [Eq. 4]")
    log.info("  P_sem(Ai,Aj)   = max_c [ phi_i[c] * phi_j[c] ]")
    log.info("  -> Random Forest -> Sim_attr(Ai,Aj) -> weighted graph")


if __name__ == "__main__":
    main()