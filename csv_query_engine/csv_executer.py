"""
csv_executer.py — Natural-language CSV query engine.

Pipeline
--------
  1. ``import_csv(path)``          → registers metadata; does NOT permanently
                                     store rows in PostgreSQL.
  2. ``query_table(name, q)``      → loads the CSV into a TEMP table for this
                                     request, runs sandboxed pandas code, then
                                     drops the temp table automatically.
  3. ``query_csv(path, q)``        → convenience wrapper (import + query).
  4. ``query_auto(q)``             → auto-detects which registered CSV the
                                     question refers to and calls query_table.

Key design change (v2):
  Rows are no longer stored permanently in PostgreSQL.  Only the file registry
  (_csv_registry) and metadata are kept.  Every query session uses
  ``CsvQuerySession`` from csv_session.py which creates a temp table,
  executes the query, and drops the table — regardless of success or failure.
  This dramatically reduces PG storage and eliminates stale data.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import chardet
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

from dotenv import load_dotenv
from groq import Groq
from loguru import logger
from sqlalchemy import create_engine, inspect, text
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr
from RestrictedPython.PrintCollector import PrintCollector

load_dotenv()


# ── CONFIG ────────────────────────────────────────────────────────────────────
# GROQ_CSV_API_KEY — dedicated key for NL→SQL translation on CSV queries.
# Falls back to GROQ_API_KEY so existing setups need no change.
GROQ_API_KEY      = os.environ.get("GROQ_CSV_API_KEY") or os.environ.get("GROQ_API_KEY")
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

PG_HOST     = os.environ.get("PG_HOST",     "localhost")
PG_PORT     = os.environ.get("PG_PORT",     "5432")
PG_DB       = os.environ.get("PG_DB",       "csvstore")
PG_USER     = os.environ.get("PG_USER",     "csvuser")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")   # empty string — override via .env

DATABASE_URL = (
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}"
    f"@{PG_HOST}:{PG_PORT}/{PG_DB}"
)


# ── GROQ CLIENT ───────────────────────────────────────────────────────────────
_groq_client = Groq(api_key=GROQ_API_KEY)


# ── POSTGRES ENGINE ───────────────────────────────────────────────────────────
def get_engine():
    """Return a SQLAlchemy engine connected to PostgreSQL."""
    return create_engine(DATABASE_URL, future=True)


# ── CSV REGISTRY (hash tracking) ─────────────────────────────────────────────

def _ensure_registry():
    """
    Create the _csv_registry table if it doesn't exist.

    v2: The registry now also stores the original file_path and column/row
    metadata so we can reconstruct df_info for code generation without loading
    the full CSV into memory.
    """
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS _csv_registry (
                table_name  TEXT PRIMARY KEY,
                file_path   TEXT,
                file_hash   TEXT NOT NULL,
                total_rows  INTEGER,
                columns_json TEXT,
                dtypes_json  TEXT,
                sample_json  TEXT,
                imported_at TIMESTAMP NOT NULL
            )
        """))
        # Migration: add new columns if they don't exist yet (idempotent)
        for col_def in [
            "file_path   TEXT",
            "total_rows  INTEGER",
            "columns_json TEXT",
            "dtypes_json  TEXT",
            "sample_json  TEXT",
        ]:
            try:
                conn.execute(text(
                    f"ALTER TABLE _csv_registry ADD COLUMN IF NOT EXISTS {col_def}"
                ))
            except Exception:
                pass  # column already exists


def _get_registry_row(table_name: str) -> dict | None:
    """Return the full registry row for a table, or None if not registered."""
    _ensure_registry()
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT file_path, file_hash, total_rows, columns_json, "
                "dtypes_json, sample_json FROM _csv_registry WHERE table_name = :t"
            ),
            {"t": table_name},
        ).fetchone()
    if not row:
        return None
    return {
        "file_path":    row[0],
        "file_hash":    row[1],
        "total_rows":   row[2],
        "columns":      json.loads(row[3]) if row[3] else [],
        "dtypes":       json.loads(row[4]) if row[4] else {},
        "sample":       json.loads(row[5]) if row[5] else [],
    }


def _get_stored_hash(table_name: str) -> str | None:
    """Return the stored MD5 hash for a table, or None if not registered."""
    row = _get_registry_row(table_name)
    return row["file_hash"] if row else None


def _save_registry(
    table_name:   str,
    file_path:    str,
    file_hash:    str,
    total_rows:   int,
    columns:      list,
    dtypes:       dict,
    sample:       list,
) -> None:
    """Upsert a registry entry — no rows are stored, only metadata."""
    _ensure_registry()
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO _csv_registry
                (table_name, file_path, file_hash, total_rows,
                 columns_json, dtypes_json, sample_json, imported_at)
            VALUES (:t, :fp, :h, :rows, :cols, :dtypes, :sample, :ts)
            ON CONFLICT (table_name) DO UPDATE
                SET file_path    = EXCLUDED.file_path,
                    file_hash    = EXCLUDED.file_hash,
                    total_rows   = EXCLUDED.total_rows,
                    columns_json = EXCLUDED.columns_json,
                    dtypes_json  = EXCLUDED.dtypes_json,
                    sample_json  = EXCLUDED.sample_json,
                    imported_at  = EXCLUDED.imported_at
        """), {
            "t":      table_name,
            "fp":     file_path,
            "h":      file_hash,
            "rows":   total_rows,
            "cols":   json.dumps(columns),
            "dtypes": json.dumps(dtypes),
            "sample": json.dumps(sample, default=str),
            "ts":     datetime.now(timezone.utc),
        })


def _compute_hash(file_path: str) -> str:
    """Compute the MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


# ── CSV → POSTGRES ────────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    name = Path(name).stem
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    if name[0].isdigit():
        name = "t_" + name
    return name


def import_csv(file_path: str, table_name: str | None = None, force: bool = False) -> str:
    """
    Register a CSV file — metadata only, no permanent row storage.

    v2 change: rows are no longer imported into PostgreSQL permanently.
    Only the schema, column types, sample rows and file hash are stored in
    the ``_csv_registry`` table.  Actual data loading happens on-demand in
    ``query_table()`` via ``CsvQuerySession``.

    Args:
        file_path  : Path to the CSV file.
        table_name : Logical name for this dataset (auto-derived from filename
                     if None).
        force      : If True, refresh metadata even if the file hash matches.

    Returns:
        The table name (logical dataset identifier).
    """
    from csv_query_engine.csv_chunker import _slugify_columns

    table_name = table_name or _slugify(file_path)

    current_hash = _compute_hash(file_path)
    stored_hash  = _get_stored_hash(table_name)

    if not force and stored_hash == current_hash:
        logger.debug(f"[IMPORT] Skipped '{file_path}' — metadata already up to date.")
        return table_name

    # Read only the header + a small sample to build metadata
    with open(file_path, "rb") as f:
        encoding = chardet.detect(f.read(65_536)).get("encoding") or "utf-8"

    # Full read for stats (avoids loading twice for large files)
    df = pd.read_csv(file_path, encoding=encoding, low_memory=False, on_bad_lines="skip")
    df = _slugify_columns(df)

    columns    = df.columns.tolist()
    dtypes     = {col: str(df[col].dtype) for col in columns}
    sample     = df.head(5).to_dict(orient="records")
    total_rows = len(df)

    _save_registry(
        table_name=table_name,
        file_path=file_path,
        file_hash=current_hash,
        total_rows=total_rows,
        columns=columns,
        dtypes=dtypes,
        sample=sample,
    )

    logger.info(f"[IMPORT] Registered '{table_name}' ({total_rows} rows, {len(columns)} cols)")
    return table_name


def import_csv_folder(folder_path: str, force: bool = False) -> list[str]:
    """
    Import every CSV in a folder — skips unchanged files automatically.

    Args:
        folder_path : Path to the folder containing CSV files.
        force       : If True, reimport all files regardless of changes.

    Returns:
        List of table names processed.
    """
    folder    = Path(folder_path)
    csv_files = list(folder.glob("*.csv"))

    if not csv_files:
        print(f"[IMPORT] No CSV files found in '{folder_path}'")
        return []

    tables = []
    for csv_file in csv_files:
        table = import_csv(str(csv_file), force=force)
        tables.append(table)

    print(f"[IMPORT] Folder scan complete — {len(tables)} file(s) processed.")
    return tables


def show_registry():
    """Print all tracked tables with their hash and last import timestamp."""
    _ensure_registry()
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT table_name, file_hash, imported_at FROM _csv_registry ORDER BY imported_at DESC")
        ).fetchall()

    if not rows:
        print("[REGISTRY] No files tracked yet.")
        return

    print(f"\n{'─'*72}")
    print(f"{'TABLE':<45} {'HASH':<12} {'IMPORTED AT'}")
    print(f"{'─'*72}")
    for row in rows:
        print(f"{row[0]:<45} {row[1][:8]}...  {row[2]}")
    print(f"{'─'*72}\n")


# ── TABLE HELPERS ─────────────────────────────────────────────────────────────

def list_tables() -> list[str]:
    """
    Return all registered dataset names from the metadata registry.

    v2: No longer reads PostgreSQL table list (rows are not permanently stored).
    Returns logical dataset names from _csv_registry instead.
    """
    try:
        _ensure_registry()
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT table_name FROM _csv_registry ORDER BY table_name")
            ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def get_table_info(table_name: str) -> dict:
    """
    Return schema metadata for a registered dataset.

    v2: Reads from _csv_registry (metadata only). No actual data rows in PG.
    """
    row = _get_registry_row(table_name)
    if not row:
        raise ValueError(f"Dataset '{table_name}' is not registered. Upload the CSV first.")
    return {
        "table":   table_name,
        "shape":   (row["total_rows"] or 0, len(row["columns"])),
        "columns": row["columns"],
        "dtypes":  row["dtypes"],
        "sample":  row["sample"],
    }


# ── TABLE AUTO-DETECTOR ───────────────────────────────────────────────────────

def detect_table(question: str) -> str | None:
    """
    Automatically detect which table the user is asking about
    based on the question and available tables in Postgres.

    Returns the table name if detected, or None if unclear.
    """
    tables = list_tables()

    if not tables:
        print("[ROUTER] No tables found in database.")
        return None

    # Build a short description of each table (columns only) to help the LLM
    tables_info = {}
    for t in tables:
        try:
            info = get_table_info(t)
            tables_info[t] = info["columns"]
        except Exception:
            tables_info[t] = []

    response = _groq_client.chat.completions.create(
        model      = GROQ_VISION_MODEL,
        messages   = [{
            "role": "user",
            "content": (
                f"Available tables and their columns:\n"
                f"{json.dumps(tables_info, indent=2)}\n\n"
                f"User question: \"{question}\"\n\n"
                f"Which table is this question about?\n"
                f"Reply ONLY with the exact table name from the list.\n"
                f"If you cannot determine the table, reply with: null"
            )
        }],
        max_tokens  = 50,
        temperature = 0.0,
    )

    content = response.choices[0].message.content
    if content is None:
        return None
    answer = content.strip().strip('"').strip("'")

    if answer.lower() == "null" or answer not in tables:
        return None
    return answer


# ── SANDBOX ───────────────────────────────────────────────────────────────────

def _safe_getitem(obj, key):
    return obj[key]


def _build_safe_globals(df: pd.DataFrame) -> dict:
    import math, statistics

    ALLOWED_BUILTINS = [
        "len", "range", "enumerate", "zip", "map", "filter",
        "sorted", "reversed", "list", "dict", "set", "tuple",
        "str", "int", "float", "bool", "round", "abs", "sum",
        "min", "max", "type", "isinstance", "repr", "any", "all",
    ]

    safe_bi = {k: getattr(builtins, k) for k in ALLOWED_BUILTINS if hasattr(builtins, k)}

    return {
        # ── RestrictedPython required guards (must use exact _name_ form) ──
        "_builtins_"             : safe_bi,
        "_print_"                : PrintCollector,       # print() → _print_()
        "_getiter_"              : iter,                 # for loops
        "_getattr_"              : safer_getattr,        # attribute access
        "_getitem_"              : _safe_getitem,        # subscript access obj[key]
        "_write_"                : lambda x: x,         # assignment targets
        "_inplacevar_"           : lambda op, x, y: x,  # += -= etc.
        "_iter_unpack_sequence_" : guarded_iter_unpack_sequence,

        # ── Available libraries inside sandbox ────────────────────────────
        "pd"        : pd,
        "math"      : math,
        "statistics": statistics,
        "df"        : df,
    }


def execute_code(code: str, df: pd.DataFrame) -> dict:
    try:
        byte_code = compile_restricted(code, filename="<llm_code>", mode="exec")
    except SyntaxError as e:
        return {"success": False, "output": "", "result": None, "error": f"Syntax error: {e}"}

    safe_env   = _build_safe_globals(df)
    local_vars = {}

    try:
        exec(byte_code, safe_env, local_vars)
    except Exception as e:
        return {"success": False, "output": "", "result": None, "error": f"{type(e).__name__}: {e}"}

    captured = local_vars.get("_print", None)
    output   = str(captured()) if callable(captured) else ""

    return {
        "success": True,
        "output" : output.strip(),
        "result" : local_vars.get("result"),
        "error"  : None,
    }


# ── CODE GENERATOR ────────────────────────────────────────────────────────────

def generate_pandas_code(question: str, df_info: dict) -> str:
    system_prompt = """You are a pandas code generator.
You receive a question about a dataframe and its schema.
Write Python code to answer the question.

Rules:
- The dataframe is always called df
- Store the final answer in a variable called result
- Use print() for intermediate steps if helpful
- Do NOT use imports — pd, math, statistics are already available
- Do NOT use os, sys, subprocess, open(), or any file operations
- Return ONLY raw Python code — no markdown, no explanation"""

    user_message = (
        f"Question: {question}\n\n"
        f"Dataframe schema:\n"
        f"  Table   : {df_info['table']}\n"
        f"  Shape   : {df_info['shape'][0]} rows x {df_info['shape'][1]} cols\n"
        f"  Columns : {json.dumps(df_info['dtypes'], indent=4)}\n"
        f"  Sample  : {json.dumps(df_info['sample'], indent=4)}\n\n"
        f"Write pandas code. Store the answer in result."
    )

    response = _groq_client.chat.completions.create(
        model      = GROQ_VISION_MODEL,
        messages   = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        max_tokens  = 1024,
        temperature = 0.0,
    )

    content = response.choices[0].message.content
    code = content.strip() if content is not None else ""

    if code.startswith("```"):
        lines = code.split("\n")
        code  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    return code


# ── ANSWER GENERATOR ──────────────────────────────────────────────────────────

def generate_answer(question: str, code: str, execution: dict) -> str:
    if execution["success"]:
        result_text = (
            f"stdout : {execution['output']}\n"
            f"result : {execution['result']}"
        )
    else:
        result_text = f"Error: {execution['error']}"

    response = _groq_client.chat.completions.create(
        model      = GROQ_VISION_MODEL,
        messages   = [{
            "role"   : "user",
            "content": (
                f"Question: {question}\n\n"
                f"Code that ran:\n{code}\n\n"
                f"Execution result:\n{result_text}\n\n"
                "Give a clear, direct, concise answer to the question "
                "based on the result. No preamble."
            ),
        }],
        max_tokens  = 512,
        temperature = 0.3,
    )

    content = response.choices[0].message.content
    return content.strip() if content is not None else ""


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def query_table(table_name: str, question: str, verbose: bool = False) -> dict:
    """
    Full pipeline: registered CSV dataset + natural-language question → answer.

    v2: Uses ``CsvQuerySession`` to load data into a temporary PostgreSQL table
    just-in-time.  The temp table is automatically dropped after each call —
    regardless of success or failure — so no permanent row storage accumulates.

    Parameters
    ----------
    table_name:
        Logical dataset name (registered via ``import_csv()``).
    question:
        Natural-language question about the dataset.
    verbose:
        Print progress messages to stdout.

    Returns
    -------
    dict with keys: question, table, code, execution, answer
    """
    from csv_query_engine.csv_session import CsvQuerySession

    if verbose:
        logger.info(f"[EXECUTOR] Table   : {table_name}")
        logger.info(f"[EXECUTOR] Question: {question}")

    # ── Step 1: Fetch metadata from registry (no data loaded yet) ─────────────
    df_info = get_table_info(table_name)
    reg_row = _get_registry_row(table_name)
    if not reg_row or not reg_row.get("file_path"):
        return {
            "question":  question,
            "table":     table_name,
            "code":      None,
            "execution": {"success": False, "output": "", "result": None,
                          "error": f"No file path registered for '{table_name}'. Re-upload the CSV."},
            "answer":    f"Dataset '{table_name}' source file is not available.",
        }

    file_path = reg_row["file_path"]
    if not Path(file_path).exists():
        return {
            "question":  question,
            "table":     table_name,
            "code":      None,
            "execution": {"success": False, "output": "", "result": None,
                          "error": f"Source file not found: {file_path}"},
            "answer":    f"The original CSV file for '{table_name}' is no longer available. Please re-upload it.",
        }

    if verbose:
        logger.info(
            f"[EXECUTOR] Shape   : {df_info['shape'][0]} rows × {df_info['shape'][1]} cols"
        )

    # ── Step 2: Generate pandas code using metadata (no data loaded yet) ──────
    code = generate_pandas_code(question, df_info)
    if verbose:
        logger.debug(f"[EXECUTOR] Code:\n{code}")

    # ── Step 3: Load data into temp table, execute, then auto-cleanup ─────────
    import uuid as _uuid
    session_id = _uuid.uuid4().hex[:16]

    with CsvQuerySession(file_path, session_id=session_id) as sess:
        df        = sess.load_as_dataframe()
        execution = execute_code(code, df)

    if verbose:
        if execution["success"]:
            logger.info(f"[EXECUTOR] Result  : {execution['result']}")
        else:
            logger.warning(f"[EXECUTOR] Error   : {execution['error']}")

    # ── Step 4: Generate natural-language answer ───────────────────────────────
    answer = generate_answer(question, code, execution)

    if verbose:
        logger.info(f"[EXECUTOR] Answer  : {answer}")

    return {
        "question":  question,
        "table":     table_name,
        "code":      code,
        "execution": execution,
        "answer":    answer,
    }


def query_auto(question: str, verbose: bool = False) -> dict:
    """
    Full pipeline with automatic table detection.

    Asks the LLM to identify which registered dataset matches the question,
    then runs ``query_table()``.
    """
    if verbose:
        logger.info(f"[ROUTER] Detecting table for: \"{question}\"")

    table = detect_table(question)

    if table is None:
        return {
            "question":  question,
            "table":     None,
            "code":      None,
            "execution": None,
            "answer":    (
                "I could not determine which dataset your question refers to. "
                f"Available tables: {list_tables()}"
            ),
        }

    if verbose:
        logger.info(f"[ROUTER] Table detected: '{table}'")

    return query_table(table, question, verbose)


def query_csv(file_path: str, question: str, verbose: bool = False) -> dict:
    """
    Convenience: register (or refresh) a CSV file and immediately query it.

    Equivalent to ``import_csv(path); query_table(name, question)``.
    """
    table_name = import_csv(file_path)
    return query_table(table_name, question, verbose)


# ── TEST ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Import folder — only new or changed CSVs will be imported
    import_csv_folder("data")

    # Show registry — see what's tracked and when
    show_registry()

    # Show available tables
    print(f"[INFO] Tables in DB: {list_tables()}")

    # ── Option A: specify table manually ─────────────────────────────────────
    # result = query_table("test", "What is the average cholesterol?")

    # ── Option B: let the LLM detect the table automatically ─────────────────
    questions = [
        "What is the average cholesterol?",
        "How many businesses are in the 2023 survey?",
        "What is the total value in the annual enterprise survey?",
    ]

    for q in questions:
        result = query_auto(q)
        print(f"\n  Q: {result['question']}")
        print(f"  Table used: {result['table']}")
        print(f"  A: {result['answer']}")