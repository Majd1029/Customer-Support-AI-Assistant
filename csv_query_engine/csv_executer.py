import sys
import os
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import hashlib
import builtins
import chardet
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone
load_dotenv()

from pathlib import Path
from groq import Groq
from sqlalchemy import create_engine, text, inspect
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr
from RestrictedPython.PrintCollector import PrintCollector


# ── CONFIG ────────────────────────────────────────────────────────────────────
# GROQ_CSV_API_KEY — dedicated key for NL→SQL translation on CSV queries.
# Falls back to GROQ_API_KEY so existing setups need no change.
GROQ_API_KEY      = os.environ.get("GROQ_CSV_API_KEY") or os.environ.get("GROQ_API_KEY")
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

PG_HOST     = os.environ.get("PG_HOST",     "localhost")
PG_PORT     = os.environ.get("PG_PORT",     "5432")
PG_DB       = os.environ.get("PG_DB",       "csvstore")
PG_USER     = os.environ.get("PG_USER",     "csvuser")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "yourpassword")

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
    Tracks MD5 hash of each imported CSV to detect updates.
    """
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS _csv_registry (
                table_name  TEXT PRIMARY KEY,
                file_hash   TEXT NOT NULL,
                imported_at TIMESTAMP NOT NULL
            )
        """))


def _get_stored_hash(table_name: str) -> str | None:
    """Return the stored MD5 hash for a table, or None if not registered."""
    _ensure_registry()
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT file_hash FROM _csv_registry WHERE table_name = :t"),
            {"t": table_name}
        ).fetchone()
    return row[0] if row else None


def _save_hash(table_name: str, file_hash: str):
    """Insert or update the hash for a table in the registry."""
    _ensure_registry()
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO _csv_registry (table_name, file_hash, imported_at)
            VALUES (:t, :h, :ts)
            ON CONFLICT (table_name) DO UPDATE
                SET file_hash   = EXCLUDED.file_hash,
                    imported_at = EXCLUDED.imported_at
        """), {"t": table_name, "h": file_hash, "ts": datetime.now(timezone.utc)})


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


def import_csv(file_path: str, table_name: str = None, force: bool = False) -> str:
    """
    Load a CSV into PostgreSQL — only if new or changed.

    Logic:
        - Table doesn't exist          -> import
        - File hash changed            -> reimport (CSV was updated)
        - Table exists & hash matches  -> skip
        - force=True                   -> always reimport

    Args:
        file_path  : Path to the CSV file.
        table_name : Postgres table name (auto-derived from filename if None).
        force      : If True, always reimport regardless of hash.

    Returns:
        The table name used.
    """
    file_path  = str(file_path)
    table_name = table_name or _slugify(file_path)

    current_hash = _compute_hash(file_path)
    stored_hash  = _get_stored_hash(table_name)
    table_exists = table_name in list_tables()

    # ── Decide whether to import ──────────────────────────────────────────────
    if not force:
        if table_exists and stored_hash == current_hash:
            print(f"[IMPORT] Skipped  '{file_path}' — no changes detected.")
            return table_name
        elif table_exists and stored_hash != current_hash:
            print(f"[IMPORT] Updated  '{file_path}' — file has changed, reimporting...")
        else:
            print(f"[IMPORT] New file '{file_path}' — importing for the first time...")

    # ── Import ────────────────────────────────────────────────────────────────
    with open(file_path, "rb") as f:
        encoding = chardet.detect(f.read(50_000)).get("encoding") or "utf-8"

    df = pd.read_csv(file_path, encoding=encoding)

    df.columns = [
        re.sub(r"[^a-z0-9]+", "", col.lower()).strip("")
        for col in df.columns
    ]

    engine = get_engine()
    df.to_sql(table_name, engine, if_exists="replace", index=False)

    _save_hash(table_name, current_hash)

    print(f"[IMPORT] Done     '{table_name}' ({len(df)} rows, {len(df.columns)} cols)")
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
    """Return all user table names (excluding internal _csv_registry table)."""
    engine = get_engine()
    inspector = inspect(engine)
    return [t for t in inspector.get_table_names() if t != "_csv_registry"]


def get_table_info(table_name: str) -> dict:
    """Return schema + 3 sample rows for a given table."""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text(f'SELECT * FROM "{table_name}" LIMIT 3'),
            conn
        )
        count_row = conn.execute(
            text(f'SELECT COUNT(*) FROM "{table_name}"')
        ).fetchone()
        total_rows = count_row[0]

    return {
        "table"  : table_name,
        "shape"  : (total_rows, len(df.columns)),
        "columns": df.columns.tolist(),
        "dtypes" : {col: str(dt) for col, dt in df.dtypes.items()},
        "sample" : df.to_dict(orient="records"),
    }


def load_table(table_name: str) -> pd.DataFrame:
    """Load an entire table from PostgreSQL into a pandas DataFrame."""
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(f'SELECT * FROM "{table_name}"'), conn)


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

    answer = response.choices[0].message.content.strip().strip('"').strip("'")

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
    output   = captured() if callable(captured) else ""

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

    code = response.choices[0].message.content.strip()

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

    return response.choices[0].message.content.strip()


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def query_table(table_name: str, question: str, verbose: bool = True) -> dict:
    """
    Full pipeline: PostgreSQL table + natural language question -> answer.
    """
    if verbose:
        print(f"\n[EXECUTOR] Table   : {table_name}")
        print(f"[EXECUTOR] Model   : {GROQ_VISION_MODEL}")
        print(f"[EXECUTOR] Question: {question}")

    df_info = get_table_info(table_name)

    if verbose:
        print(f"[EXECUTOR] Shape   : {df_info['shape'][0]} rows x {df_info['shape'][1]} cols")
        print(f"[EXECUTOR] Generating code...")

    code = generate_pandas_code(question, df_info)

    if verbose:
        print(f"[EXECUTOR] Code:\n{'─'*50}\n{code}\n{'─'*50}")
        print(f"[EXECUTOR] Loading table from Postgres...")

    df = load_table(table_name)

    if verbose:
        print(f"[EXECUTOR] Running in sandbox...")

    execution = execute_code(code, df)

    if verbose:
        if execution["success"]:
            print(f"[EXECUTOR] Result  : {execution['result']}")
        else:
            print(f"[EXECUTOR] Error   : {execution['error']}")

    answer = generate_answer(question, code, execution)

    if verbose:
        print(f"[EXECUTOR] Answer  : {answer}")

    return {
        "question" : question,
        "table"    : table_name,
        "code"     : code,
        "execution": execution,
        "answer"   : answer,
    }


def query_auto(question: str, verbose: bool = True) -> dict:
    """
    Full pipeline with automatic table detection.
    The user doesn't need to specify the table name — the LLM picks it.

    Returns same dict as query_table(), plus 'table' key showing which was used.
    """
    if verbose:
        print(f"\n[ROUTER] Detecting table for question: \"{question}\"")

    table = detect_table(question)

    if table is None:
        if verbose:
            print(f"[ROUTER] Could not determine table. Available: {list_tables()}")
        return {
            "question" : question,
            "table"    : None,
            "code"     : None,
            "execution": None,
            "answer"   : (
                "I could not determine which dataset your question refers to. "
                f"Available tables: {list_tables()}"
            ),
        }

    if verbose:
        print(f"[ROUTER] Table detected: '{table}'")

    return query_table(table, question, verbose)


def query_csv(file_path: str, question: str, verbose: bool = True) -> dict:
    """
    Convenience wrapper: auto-imports/updates CSV in Postgres if needed,
    then runs query_table() on it.
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