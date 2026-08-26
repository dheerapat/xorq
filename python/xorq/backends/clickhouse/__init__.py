from __future__ import annotations

import contextlib
from typing import Any, Mapping

import pyarrow as pa
import sqlglot as sg
import sqlglot.expressions as sge

import xorq.vendor.ibis.expr.schema as sch
from xorq.backends.clickhouse.compiler import compiler
from xorq.vendor.ibis.backends.sql import SQLBackend
from xorq.vendor.ibis.expr import types as ir


__all__ = ["Backend", "connect"]


class _ClickHouseCursor:
    """Thin DB-API-like wrapper around clickhouse-connect result."""

    def __init__(self, result, schema: sch.Schema | None = None):
        self._result = result
        self._schema = schema
        # clickhouse-connect result: result.result_rows, result.column_names
        # fallback: DB-API cursor with fetchall
        self.description = None
        if hasattr(result, "column_names"):
            cols = result.column_names
            # DB-API description is 7-tuple per column
            self.description = [(c, None, None, None, None, None, None) for c in cols]
        elif hasattr(result, "description"):
            self.description = result.description

        self._rows = None
        self._idx = 0

    def fetchall(self):
        if self._rows is not None:
            return self._rows
        if hasattr(self._result, "result_rows"):
            return self._result.result_rows
        if hasattr(self._result, "fetchall"):
            return self._result.fetchall()
        return []

    def fetchmany(self, size: int):
        rows = self.fetchall()
        batch = rows[self._idx : self._idx + size]
        self._idx += size
        return batch

    def fetchone(self):
        rows = self.fetchall()
        if self._idx >= len(rows):
            return None
        row = rows[self._idx]
        self._idx += 1
        return row

    def close(self):
        if hasattr(self._result, "close"):
            try:
                self._result.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Backend(SQLBackend):
    name = "clickhouse"
    compiler = compiler
    dialect = "clickhouse"

    _top_level_methods = ("connect_env",)
    _secret_keys = ("password",)

    def do_connect(
        self,
        host: str = "localhost",
        port: int = 8123,
        database: str = "default",
        user: str | None = None,
        username: str | None = None,
        password: str = "",
        secure: bool = False,
        **kwargs: Any,
    ) -> None:
        """Connect to ClickHouse.

        Parameters
        ----------
        host
            Host name.
        port
            HTTP port (8123 plain, 8443 secure).
        database
            Database name.
        user / username
            User (alias).
        password
            Password.
        secure
            Use HTTPS.
        **kwargs
            Forwarded to ``clickhouse_connect.get_client``.

        Requires ``pip install "xorq[clickhouse]"`` (``clickhouse-connect``).
        """
        try:
            import clickhouse_connect  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "clickhouse-connect is required for the ClickHouse backend. "
                'Install with `pip install "xorq[clickhouse]"` or `pip install clickhouse-connect`.'
            ) from e

        # clickhouse_connect uses `username`, ibis historically uses `user`
        if username is None and user is not None:
            username = user
        if username is None:
            username = "default"

        self._host = host
        self._port = port
        self._database = database
        self._username = username
        self._password = password
        self._secure = secure
        self._connect_kwargs = kwargs

        self.con = clickhouse_connect.get_client(
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            secure=secure,
            **kwargs,
        )

    def disconnect(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        if not isinstance(query, str):
            query = query.sql(dialect=self.dialect)
        # clickhouse_connect
        result = self.con.query(query, **kwargs)
        return _ClickHouseCursor(result)

    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        with contextlib.closing(self.raw_sql(*args, **kwargs)) as cur:
            yield cur

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        # ponytail: minimal— run query with LIMIT 0 and infer from Arrow
        # ClickHouse supports DESCRIBE but LIMIT 0 is dialect-agnostic
        limited = f"SELECT * FROM ({query}) AS _t LIMIT 0"
        try:
            result = self.con.query(limited)
            # try arrow
            if hasattr(result, "to_arrow_table"):
                arrow_table = result.to_arrow_table()
                return sch.Schema.from_pyarrow(arrow_table.schema)
            # fallback: column_names + column_types
            if hasattr(result, "column_names"):
                # infer types via ClickHouse type mapper from empty result is hard;
                # fallback to string
                import xorq.vendor.ibis.expr.datatypes as dt

                return sch.Schema(
                    {name: dt.string for name in result.column_names}
                )
        except Exception:
            pass
        # fallback: ask ClickHouse DESCRIBE
        # parse table name from query is fragile, just return empty
        raise NotImplementedError("Cannot infer schema for query: " + query[:200])

    def get_schema(
        self,
        table_name: str,
        *,
        catalog: str | None = None,
        database: str | None = None,
    ) -> sch.Schema:
        # ClickHouse DESCRIBE TABLE
        ident = sg.table(
            table_name, db=database or self._database, quoted=self.compiler.quoted
        ).sql(dialect=self.dialect)
        # DESCRIBE returns (name, type, default_type, default_expression, comment, codec_expression, ttl_expression)
        result = self.con.query(f"DESCRIBE TABLE {ident}")
        rows = result.result_rows if hasattr(result, "result_rows") else result.fetchall()
        type_mapper = self.compiler.type_mapper
        fields = {}
        for row in rows:
            col_name, col_type = row[0], row[1]
            fields[col_name] = type_mapper.from_string(col_type, nullable=True)
        return sch.Schema(fields)

    def list_tables(
        self, *, like: str | None = None, database: str | None = None
    ) -> list[str]:
        db = database or self._database
        result = self.con.query(f"SHOW TABLES FROM {sg.to_identifier(db, quoted=self.compiler.quoted).sql(dialect=self.dialect)}")
        rows = result.result_rows if hasattr(result, "result_rows") else result.fetchall()
        tables = [r[0] for r in rows]
        return self._filter_with_like(tables, like)

    def list_databases(self, *, like: str | None = None) -> list[str]:
        result = self.con.query("SHOW DATABASES")
        rows = result.result_rows if hasattr(result, "result_rows") else result.fetchall()
        dbs = [r[0] for r in rows]
        return self._filter_with_like(dbs, like)

    @property
    def version(self) -> str:
        result = self.con.query("SELECT version()")
        rows = result.result_rows if hasattr(result, "result_rows") else result.fetchall()
        return rows[0][0] if rows else "unknown"

    @property
    def current_database(self) -> str:
        return self._database

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **_: Any,
    ) -> pa.ipc.RecordBatchReader:
        self._run_pre_execute_hooks(expr)
        sql = self.compile(expr, params=params, limit=limit)
        result = self.con.query(sql)
        # native arrow path (clickhouse-connect >=0.8 with use_arrow)
        if hasattr(result, "to_arrow_table"):
            table = result.to_arrow_table()
            if isinstance(table, pa.Table):
                return table.to_reader()
            # fallback: pandas-like
            import pandas as pd  # noqa: PLC0415

            df = table.to_pandas() if hasattr(table, "to_pandas") else pd.DataFrame(table)
            return pa.Table.from_pandas(df, preserve_index=False).to_reader()
        if hasattr(result, "result_rows"):
            import pandas as pd  # noqa: PLC0415

            df = pd.DataFrame(result.result_rows, columns=result.column_names)
            return pa.Table.from_pandas(df, preserve_index=False).to_reader()
        raise RuntimeError("Unsupported clickhouse-connect result type")

    def to_pyarrow(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Table:
        return self.to_pyarrow_batches(expr, params=params, limit=limit, **kwargs).read_all()

    def execute(
        self,
        expr: ir.Expr,
        params: Mapping | None = None,
        limit: str | None = "default",
        **kwargs: Any,
    ) -> Any:
        table = self.to_pyarrow(expr, params=params, limit=limit, **kwargs)
        return expr.__pandas_result__(table.to_pandas())

    def read_record_batches(
        self,
        record_batches: pa.RecordBatchReader | pa.Table,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Ingest Arrow data into ClickHouse via ``insert_arrow``."""
        from xorq.vendor.ibis.util import gen_name  # noqa: PLC0415

        table_name = table_name or gen_name("clickhouse_memtable")
        if isinstance(record_batches, pa.RecordBatchReader):
            table = record_batches.read_all()
        else:
            table = record_batches
        # ponytail: minimal— create table implicitly via insert; ClickHouse will create if not exists with MergeTree
        # Use client.insert_arrow if available, else fallback to insert
        if hasattr(self.con, "insert_arrow"):
            self.con.insert_arrow(table_name, table)
        elif hasattr(self.con, "insert"):
            # convert to pandas for generic insert
            self.con.insert(table_name, table.to_pandas())
        else:
            raise RuntimeError("ClickHouse client does not support insert_arrow/insert")
        return self.table(table_name)

    @classmethod
    def connect_env(cls, **kwargs: Any):
        from xorq.common.utils.clickhouse_utils import make_connection  # noqa: PLC0415

        return make_connection(**kwargs)


def connect(*args: Any, **kwargs: Any) -> Backend:
    con = Backend()
    return con.connect(*args, **kwargs)
