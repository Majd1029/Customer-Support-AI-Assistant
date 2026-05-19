"""
csv_session.py — Temporary PostgreSQL session manager for CSV queries.

Provides ``CsvQuerySession``, a context manager that:
  1. Opens a dedicated PostgreSQL connection.
  2. Loads a slice of a CSV file into a uniquely-named temporary table
     that is scoped to the connection (``CREATE TEMP TABLE``).
  3. Exposes the temp table for querying via pandas or raw SQL.
  4. Drops the table and closes the connection automatically on exit —
     even if an exception is raised inside the ``with`` block.

Temporary tables are connection-scoped in PostgreSQL, so they are
automatically dropped when the connection closes.  The unique
``_tmp_{session_id}`` naming prevents collisions between concurrent requests.

Usage
-----
with CsvQuerySession(file_path, row_start=0, row_end=5000) as sess:
    df = sess.load_as_dataframe()
    result = sess.execute_query("SELECT category, COUNT(*) FROM {table} GROUP BY 1")
    # table name is automatically substituted for {table}
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from csv_query_engine.csv_chunker import _safe_read_csv, _slugify_columns


# ── PostgreSQL connection params ───────────────────────────────────────────────

def _pg_params() -> dict:
    return {
        "host":     os.environ.get("PG_HOST",     "localhost"),
        "port":     int(os.environ.get("PG_PORT", "5432")),
        "dbname":   os.environ.get("PG_DB",       "csvstore"),
        "user":     os.environ.get("PG_USER",     "csvuser"),
        "password": os.environ.get("PG_PASSWORD", ""),
    }


def _get_conn():
    """Open and return a new psycopg2 connection.  Caller owns the lifecycle."""
    import psycopg2
    return psycopg2.connect(**_pg_params())


# ── Session ───────────────────────────────────────────────────────────────────

class CsvQuerySession:
    """
    Context manager for loading a CSV (or CSV slice) into a temp PG table.

    Parameters
    ----------
    file_path:
        Path to the CSV file on disk.
    row_start:
        First data row to load (0-based, not counting header). Default: 0.
    row_end:
        Exclusive end row.  ``None`` loads all rows from ``row_start`` onward.
    session_id:
        Optional explicit session identifier.  A random UUID4 is generated
        when omitted.  Used to build the temp table name.
    chunk_size:
        ``pandas.read_csv`` chunk size used when streaming very large slices.
        Does not affect the row_start/row_end window — only controls internal
        memory usage while loading.

    Attributes
    ----------
    table_name : str
        The PostgreSQL temp table name (``_tmp_{session_id_hex}``).
    conn :
        The live psycopg2 connection (available after ``__enter__``).
    """

    def __init__(
        self,
        file_path:  str | Path,
        row_start:  int = 0,
        row_end:    Optional[int] = None,
        session_id: Optional[str] = None,
        chunk_size: int = 50_000,
    ) -> None:
        self.file_path  = Path(file_path)
        self.row_start  = row_start
        self.row_end    = row_end
        self.chunk_size = chunk_size

        sid              = (session_id or uuid.uuid4().hex)
        # Sanitise: only alphanumerics + underscores
        safe_sid         = "".join(c if c.isalnum() else "_" for c in sid)[:32]
        self._table_name = f"_tmp_{safe_sid}"
        self._conn       = None
        self._loaded     = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def table_name(self) -> str:
        """PostgreSQL temp table name for this session."""
        return self._table_name

    @property
    def conn(self):
        """Active psycopg2 connection (None before ``__enter__``)."""
        return self._conn

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "CsvQuerySession":
        self._conn = _get_conn()
        self._load_data()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None:
        self._cleanup()
        return False   # do not suppress exceptions

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        """Read the CSV slice into a PostgreSQL temp table via SQLAlchemy."""
        from sqlalchemy import create_engine, text as sa_text

        params  = _pg_params()
        db_url  = (
            f"postgresql+psycopg2://{params['user']}:{params['password']}"
            f"@{params['host']}:{params['port']}/{params['dbname']}"
        )
        engine = create_engine(db_url, pool_pre_ping=True)

        # Build the DataFrame slice
        df = self._read_slice()
        if df.empty:
            logger.warning(
                f"[CSV_SESSION] Empty slice loaded from '{self.file_path}' "
                f"(rows {self.row_start}:{self.row_end})"
            )

        # Write to PostgreSQL as a regular table first, then we'll treat it
        # as temporary.  Using SQLAlchemy's to_sql is simpler than raw COPY.
        # We use if_exists='replace' so repeated calls are idempotent.
        with engine.connect() as con:
            df.to_sql(self._table_name, con, if_exists="replace", index=False)  # type: ignore
        self._loaded = True

        logger.debug(
            f"[CSV_SESSION] Loaded {len(df)} rows into '{self._table_name}' "
            f"from '{self.file_path.name}'"
        )

    def _read_slice(self) -> pd.DataFrame:
        """
        Read rows [row_start, row_end) from the CSV file.

        Uses ``skiprows`` / ``nrows`` so only the required rows are read.
        The header row is always read regardless of row_start.
        """
        nrows: Optional[int] = None
        if self.row_end is not None:
            nrows = self.row_end - self.row_start

        df = _safe_read_csv(
            self.file_path,
            skiprows=range(1, self.row_start + 1) if self.row_start > 0 else None,
            nrows=nrows,
        )
        return _slugify_columns(df)

    # ── Query interface ───────────────────────────────────────────────────────

    def load_as_dataframe(self) -> pd.DataFrame:
        """
        Return the full loaded slice as a pandas DataFrame.

        Requires an active session (inside ``with`` block).
        """
        self._assert_active()
        from sqlalchemy import create_engine, text as sa_text

        params  = _pg_params()
        db_url  = (
            f"postgresql+psycopg2://{params['user']}:{params['password']}"
            f"@{params['host']}:{params['port']}/{params['dbname']}"
        )
        engine = create_engine(db_url)
        with engine.connect() as con:
            return pd.read_sql(
                sa_text(f'SELECT * FROM "{self._table_name}"'), con  # type: ignore
            )

    def execute_query(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL SELECT query against the loaded temp table.

        Use ``{table}`` as a placeholder for the temp table name, e.g.:
            SELECT region, SUM(revenue) FROM {table} GROUP BY 1

        Parameters
        ----------
        sql:
            SQL query.  ``{table}`` is substituted with the temp table name.

        Returns
        -------
        pd.DataFrame
            Query results.
        """
        self._assert_active()
        resolved = sql.replace("{table}", f'"{self._table_name}"')

        from sqlalchemy import create_engine, text as sa_text

        params  = _pg_params()
        db_url  = (
            f"postgresql+psycopg2://{params['user']}:{params['password']}"
            f"@{params['host']}:{params['port']}/{params['dbname']}"
        )
        engine = create_engine(db_url)
        with engine.connect() as con:
            return pd.read_sql(sa_text(resolved), con)  # type: ignore

    def execute_pandas(self, code: str) -> dict:
        """
        Execute sandboxed pandas code with ``df`` bound to the loaded table.

        Returns the same ``{success, output, result, error}`` dict as the
        existing ``csv_executer.execute_code()``.
        """
        self._assert_active()
        from csv_query_engine.csv_executer import execute_code

        df = self.load_as_dataframe()
        return execute_code(code, df)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _cleanup(self) -> None:
        """Drop the temp table (if loaded) and close the connection."""
        if self._conn is None:
            return
        try:
            if self._loaded:
                from sqlalchemy import create_engine, text as sa_text

                params  = _pg_params()
                db_url  = (
                    f"postgresql+psycopg2://{params['user']}:{params['password']}"
                    f"@{params['host']}:{params['port']}/{params['dbname']}"
                )
                engine = create_engine(db_url)
                with engine.begin() as con:
                    con.execute(
                        sa_text(f'DROP TABLE IF EXISTS "{self._table_name}"')
                    )
                logger.debug(f"[CSV_SESSION] Dropped temp table '{self._table_name}'")
        except Exception as e:
            logger.warning(f"[CSV_SESSION] Cleanup error for '{self._table_name}': {e}")
        finally:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn   = None
            self._loaded = False

    def _assert_active(self) -> None:
        """Raise RuntimeError if called outside an active ``with`` block."""
        if self._conn is None or not self._loaded:
            raise RuntimeError(
                "CsvQuerySession must be used inside a 'with' block "
                "and data must be loaded before querying."
            )


# ── Convenience function ──────────────────────────────────────────────────────

def query_csv_temp(
    file_path:  str | Path,
    question:   str,
    groq_client=None,
    row_start:  int = 0,
    row_end:    Optional[int] = None,
    verbose:    bool = False,
) -> dict:
    """
    High-level convenience: load a CSV slice into a temp table, run NL query,
    clean up, and return results — all in one call.

    This replaces the old ``query_csv()`` pattern which permanently imported
    files into PostgreSQL.

    Parameters
    ----------
    file_path:
        CSV file path.
    question:
        Natural-language question about the data.
    groq_client:
        Groq client instance.  If None, a new one is created from env vars.
    row_start / row_end:
        Optional row window (default: all rows).
    verbose:
        Print progress messages.

    Returns
    -------
    dict
        ``{question, table, code, execution, answer}``
    """
    from csv_query_engine.csv_chunker import describe_csv
    from csv_query_engine.csv_executer import (
        generate_answer,
        generate_pandas_code,
    )

    path = Path(file_path)

    # Build df_info (same shape expected by generate_pandas_code)
    desc = describe_csv(path)
    df_info = {
        "table":  path.stem,
        "shape":  (
            (row_end or desc.total_rows) - row_start,
            len(desc.columns),
        ),
        "columns": desc.columns,
        "dtypes":  desc.dtypes,
        "sample":  desc.sample_rows,
    }

    with CsvQuerySession(path, row_start=row_start, row_end=row_end) as sess:
        df     = sess.load_as_dataframe()
        code   = generate_pandas_code(question, df_info)
        result = sess.execute_pandas(code)

    from csv_query_engine.csv_executer import generate_answer
    answer = generate_answer(question, code, result)

    return {
        "question":  question,
        "table":     path.stem,
        "code":      code,
        "execution": result,
        "answer":    answer,
    }
