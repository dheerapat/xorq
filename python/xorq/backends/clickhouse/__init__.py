from __future__ import annotations

import contextlib
import re
import warnings
from typing import TYPE_CHECKING, Any, Mapping
from urllib.parse import parse_qs, urlsplit

import pyarrow as pa
import sqlglot as sg


if TYPE_CHECKING:
    import pandas as pd

import xorq.vendor.ibis.expr.schema as sch
from xorq.backends.clickhouse.compiler import compiler
from xorq.vendor.ibis.backends.sql import SQLBackend
from xorq.vendor.ibis.expr import types as ir


__all__ = ["Backend", "connect"]


# Arrow -> ClickHouse type mapping used to CREATE TABLE from ingested Arrow data.
# The vendored ibis ClickHouse type mapper's `from_ibis` is incompatible with the
# pinned sqlglot (it references a removed `NULLABLE` typecode), so we map pyarrow
# types directly for DDL generation instead of going through the broken mapper.
_ARROW_UNIT_SCALE = {"s": 0, "ms": 3, "us": 6, "ns": 9}


def _arrow_is_nested(t: pa.DataType) -> bool:
    return (
        pa.types.is_list(t)
        or pa.types.is_large_list(t)
        or pa.types.is_map(t)
        or pa.types.is_struct(t)
    )


def _arrow_type_to_clickhouse(t: pa.DataType) -> str:
    if pa.types.is_int8(t):
        return "Int8"
    if pa.types.is_int16(t):
        return "Int16"
    if pa.types.is_int32(t):
        return "Int32"
    if pa.types.is_int64(t):
        return "Int64"
    if pa.types.is_uint8(t):
        return "UInt8"
    if pa.types.is_uint16(t):
        return "UInt16"
    if pa.types.is_uint32(t):
        return "UInt32"
    if pa.types.is_uint64(t):
        return "UInt64"
    if pa.types.is_float16(t):
        return "Float32"
    if pa.types.is_float32(t):
        return "Float32"
    if pa.types.is_float64(t):
        return "Float64"
    if pa.types.is_boolean(t):
        return "Bool"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "String"
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return "String"
    if pa.types.is_date(t) or pa.types.is_date32(t) or pa.types.is_date64(t):
        return "Date"
    if pa.types.is_timestamp(t):
        return f"DateTime64({_ARROW_UNIT_SCALE.get(t.unit, 3)})"
    if pa.types.is_decimal(t):
        return f"Decimal({t.precision},{t.scale})"
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return f"Array({_arrow_type_to_clickhouse(t.value_type)})"
    if pa.types.is_map(t):
        return f"Map({_arrow_type_to_clickhouse(t.key_type)}, {_arrow_type_to_clickhouse(t.item_type)})"
    if pa.types.is_struct(t):
        inner = ", ".join(f"{f.name} {_arrow_type_to_clickhouse(f.type)}" for f in t)
        return f"Tuple({inner})"
    if pa.types.is_dictionary(t):
        return _arrow_type_to_clickhouse(t.value_type)
    return "String"


def _ch_default_for_base(base: str) -> str | None:
    """Return a DEFAULT literal for a ClickHouse base type, or None if no sensible default.

    Per `schema-types-avoid-nullable`: use DEFAULT ''/0 instead of Nullable when
    null semantics aren't required.
    """
    if base.startswith("LowCardinality("):
        # LowCardinality(String) / LowCardinality(Nullable(String)) handled upstream; unwrap
        inner = base[len("LowCardinality(") : -1]
        # Recurse on inner for default, but LowCardinality(String) default is still ''
        if "String" in inner:
            return "''"
        return _ch_default_for_base(inner)
    if base == "String":
        return "''"
    if base == "Bool":
        return "false"
    if base.startswith("Int") or base.startswith("UInt"):
        return "0"
    if base.startswith("Float"):
        return "0"
    if base == "Date":
        return "toDate(0)"
    if base.startswith("DateTime64"):
        m = re.search(r"DateTime64\((\d+)", base)
        scale = m.group(1) if m else "3"
        return f"toDateTime64(0, {scale})"
    if base.startswith("DateTime"):
        return "toDateTime(0)"
    if base.startswith("Decimal"):
        return "0"
    # Arrays/Maps/Tuples: no default needed; ClickHouse defaults to empty
    return None


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

    # Scan/run-time caps applied to read queries (agent-query-safety). LIMIT is the
    # caller's responsibility via compile(); these are the real guardrails.
    # Per `agent-query-safety`: max_rows_to_read/max_bytes_to_read are the scan caps,
    # max_execution_time + timeout_before... is wall-clock, plus estimate + result caps.
    _READ_SETTINGS = {
        "max_execution_time": 30,
        "max_rows_to_read": 1_000_000_000,
        "max_bytes_to_read": 100_000_000_000,
        "timeout_before_checking_execution_speed": 0,
        "max_estimated_execution_time": 60,
        "max_result_rows": 10000,
        "result_overflow_mode": "break",
    }

    def raw_sql(self, query: str | sg.Expression, **kwargs: Any) -> Any:
        if not isinstance(query, str):
            query = query.sql(dialect=self.dialect)
        # Cap scans/run time for read queries (agent-query-safety).
        # Strip leading comments/whitespace so "-- comment\nSELECT" still gets capped.
        stripped = re.sub(
            r"^\s*(?:--[^\n]*\n|\s|/\*.*?\*/)*", "", query, flags=re.DOTALL
        ).lstrip()
        if stripped.upper().startswith(("SELECT", "WITH", "EXPLAIN", "DESCRIBE")):
            # Only SELECT/WITH need scan caps; EXPLAIN/DESCRIBE are cheap but also safe to cap
            needs_cap = stripped.upper().startswith(("SELECT", "WITH"))
            if needs_cap:
                settings = kwargs.setdefault("settings", {})
                for k, v in self._READ_SETTINGS.items():
                    settings.setdefault(k, v)
        result = self.con.query(query, **kwargs)
        return _ClickHouseCursor(result)

    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        with contextlib.closing(self.raw_sql(*args, **kwargs)) as cur:
            yield cur

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        # Infer types from an Arrow result over a zero-row subquery. ClickHouse
        # returns correct column types even with no rows, so we avoid the
        # all-String fallback (schema-types-native-types).
        limited = f"SELECT * FROM ({query}) AS _t LIMIT 0"
        try:
            arrow_table = self.con.query_arrow(limited)
            return sch.Schema.from_pyarrow(arrow_table.schema)
        except Exception:
            pass
        # Fallback: DESCRIBE the SELECT to recover (name, type) pairs and map
        # them back to ibis types via the (working) `from_string` path.
        result = self.con.query(f"DESCRIBE {query}")
        rows = (
            result.result_rows if hasattr(result, "result_rows") else result.fetchall()
        )
        return sch.Schema(
            {
                name: self.compiler.type_mapper.from_string(col_type, nullable=True)
                for name, col_type in ((r[0], r[1]) for r in rows)
            }
        )

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
        rows = (
            result.result_rows if hasattr(result, "result_rows") else result.fetchall()
        )
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
        result = self.con.query(
            f"SHOW TABLES FROM {sg.to_identifier(db, quoted=self.compiler.quoted).sql(dialect=self.dialect)}"
        )
        rows = (
            result.result_rows if hasattr(result, "result_rows") else result.fetchall()
        )
        tables = [r[0] for r in rows]
        return self._filter_with_like(tables, like)

    def list_databases(self, *, like: str | None = None) -> list[str]:
        result = self.con.query("SHOW DATABASES")
        rows = (
            result.result_rows if hasattr(result, "result_rows") else result.fetchall()
        )
        dbs = [r[0] for r in rows]
        return self._filter_with_like(dbs, like)

    # --- agent-discovery-schema helpers (system.*) ---
    def get_sorting_key(
        self, table_name: str, *, database: str | None = None
    ) -> dict[str, str]:
        """Return sorting_key / primary_key / partition_key for a table.

        Per `agent-discovery-schema` step 4: filtering on sort key allows granule skipping.
        """
        db = database or self._database
        q = (
            "SELECT sorting_key, primary_key, partition_key, engine "
            "FROM system.tables WHERE database = {db:String} AND name = {tbl:String}"
        )
        result = self.con.query(q, parameters={"db": db, "tbl": table_name})
        rows = (
            result.result_rows if hasattr(result, "result_rows") else result.fetchall()
        )
        if not rows:
            return {
                "sorting_key": "",
                "primary_key": "",
                "partition_key": "",
                "engine": "",
            }
        # columns: sorting_key, primary_key, partition_key, engine
        return dict(
            zip(["sorting_key", "primary_key", "partition_key", "engine"], rows[0])
        )

    def get_skipping_indices(
        self, table_name: str, *, database: str | None = None
    ) -> list[dict[str, Any]]:
        """Return data skipping indices for a table (step 5 of discovery).

        Per `query-index-skipping-indices`: bloom_filter / set / minmax / tokenbf.
        """
        db = database or self._database
        q = (
            "SELECT name, type, expr, granularity "
            "FROM system.data_skipping_indices WHERE database = {db:String} AND table = {tbl:String}"
        )
        try:
            result = self.con.query(q, parameters={"db": db, "tbl": table_name})
        except Exception:
            return []
        rows = (
            result.result_rows if hasattr(result, "result_rows") else result.fetchall()
        )
        cols = ["name", "type", "expr", "granularity"]
        # clickhouse-connect may return result.column_names
        if hasattr(result, "column_names") and result.column_names:
            cols = list(result.column_names)
        return [dict(zip(cols, r)) for r in rows]

    def get_columns_with_comments(
        self, table_name: str, *, database: str | None = None
    ) -> list[dict[str, Any]]:
        """Return columns with types and comments (step 3 of discovery)."""
        db = database or self._database
        q = (
            "SELECT name, type, comment, position "
            "FROM system.columns WHERE database = {db:String} AND table = {tbl:String} ORDER BY position"
        )
        result = self.con.query(q, parameters={"db": db, "tbl": table_name})
        rows = (
            result.result_rows if hasattr(result, "result_rows") else result.fetchall()
        )
        cols = ["name", "type", "comment", "position"]
        if hasattr(result, "column_names") and result.column_names:
            cols = list(result.column_names)
        return [dict(zip(cols, r)) for r in rows]

    @property
    def version(self) -> str:
        result = self.con.query("SELECT version()")
        rows = (
            result.result_rows if hasattr(result, "result_rows") else result.fetchall()
        )
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
        # Native Arrow read (clickhouse-connect >=0.8 query_arrow). Adds safety
        # scan caps (agent-query-safety) without forcing a row LIMIT.
        if hasattr(self.con, "query_arrow"):
            table = self.con.query_arrow(sql, settings=self._READ_SETTINGS)
            if isinstance(table, pa.Table):
                return table.to_reader()
        result = self.con.query(sql)
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
        return self.to_pyarrow_batches(
            expr, params=params, limit=limit, **kwargs
        ).read_all()

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
        *,
        order_by: str = "tuple()",
        partition_by: str | None = None,
        low_cardinality_cols: list[str] | set[str] | None = None,
        nullable_cols: set[str] | list[str] | bool | None = None,
        skipping_indices: list[str] | None = None,
        chunk_size: int = 100_000,
        **kwargs: Any,
    ) -> ir.Table:
        """Ingest Arrow data into a ClickHouse table.

        ClickHouse ``INSERT`` does not auto-create tables, so we create an
        explicit ``MergeTree`` table from the Arrow schema first.

        Per `schema-pk-plan-before-creation` ORDER BY is immutable — choose it before
        creation based on query patterns. `tuple()` means no sparse index (full scan).
        Pass low-cardinality string columns via `low_cardinality_cols` (per
        `schema-types-lowcardinality`) and control nullability via `nullable_cols`
        (per `schema-types-avoid-nullable`).

        Parameters
        ----------
        order_by
            ORDER BY expression. Default ``tuple()`` = no primary index. For real
            workloads pass e.g. ``"event_type, toDate(timestamp), user_id"`` with
            low-cardinality columns first (per `schema-pk-cardinality-order`).
        partition_by
            Optional PARTITION BY expression (e.g. ``"toYYYYMM(timestamp)"``).
            Per `schema-partition-*` keep cardinality <1k and prefer monthly over
            daily. ``None`` = no partitioning (recommended to start per
            `schema-partition-start-without`).
        low_cardinality_cols
            Column names that should be ``LowCardinality(String)`` (<10k uniques).
        nullable_cols
            ``None`` = auto (Nullable only if Arrow field is nullable AND data contains nulls).
            ``True``/``False`` = force all/none. ``set`` = allowlist.
        skipping_indices
            Raw ``INDEX name expr TYPE type GRANULARITY N`` clauses (per
            `query-index-skipping-indices`).
        """
        from xorq.vendor.ibis.util import gen_name  # noqa: PLC0415

        if order_by == "tuple()":
            warnings.warn(
                "ORDER BY tuple() disables the sparse primary index (full scans). "
                "Pass `order_by` matching your filter columns; see schema-pk-plan-before-creation. "
                "Immutable after creation — requires table migration to fix.",
                UserWarning,
                stacklevel=2,
            )

        table_name = table_name or gen_name("clickhouse_memtable")
        table = (
            record_batches.read_all()
            if isinstance(record_batches, pa.RecordBatchReader)
            else record_batches
        )

        # Per `schema-types-avoid-nullable` and `allow_nullable_key` restriction:
        # ORDER BY / PARTITION BY columns must be non-nullable. Force them to non-nullable
        # even if caller asked for Nullable or data contains nulls.
        raw_keys = f"{order_by} {partition_by or ''}"
        key_cols = {f.name for f in table.schema if f.name in raw_keys}

        columns = ", ".join(
            self._arrow_field_to_clickhouse(
                f.name,
                f,
                table=table,
                low_cardinality_cols=(
                    set(low_cardinality_cols) if low_cardinality_cols else None
                ),
                nullable_cols=(
                    set(nullable_cols)
                    if isinstance(nullable_cols, (list, set, tuple))
                    else nullable_cols
                ),
                force_non_nullable=(f.name in key_cols),
            )
            for f in table.schema
        )
        idx_sql = ""
        if skipping_indices:
            idx_sql = ", " + ", ".join(skipping_indices)
        part_sql = f" PARTITION BY {partition_by}" if partition_by else ""
        # Normalize ORDER BY: multi-column needs parentheses, single col works either way
        ob = order_by.strip()
        if ob != "tuple()" and not (ob.startswith("(") and ob.endswith(")")):
            ob = f"({ob})"
        ddl = (
            f"CREATE TABLE IF NOT EXISTS {table_name} ({columns}{idx_sql}) "
            f"ENGINE = MergeTree{part_sql} ORDER BY {ob}"
        )
        self.con.command(ddl)

        # Batch inserts (insert-batch-size: ~100K rows per part).
        for start in range(0, table.num_rows, chunk_size):
            self.con.insert_arrow(table_name, table.slice(start, chunk_size))

        return self.table(table_name)

    @staticmethod
    def _arrow_field_to_clickhouse(
        name: str,
        field: pa.Field,
        *,
        table: pa.Table | None = None,
        low_cardinality_cols: set[str] | None = None,
        nullable_cols: set[str] | bool | None = None,
        force_non_nullable: bool = False,
    ) -> str:
        base = _arrow_type_to_clickhouse(field.type)
        is_nested = _arrow_is_nested(field.type)

        # LowCardinality handling (schema-types-lowcardinality): only String benefits
        if low_cardinality_cols and name in low_cardinality_cols and base == "String":
            base = "LowCardinality(String)"

        # Determine nullability (schema-types-avoid-nullable)
        if force_non_nullable:
            nullable = False
        elif is_nested:
            nullable = False
        else:
            if isinstance(nullable_cols, bool):
                nullable = field.nullable and nullable_cols
            elif isinstance(nullable_cols, (set, list, tuple)):
                nullable = name in nullable_cols and field.nullable
            else:  # None = auto
                if table is not None:
                    try:
                        col = table.column(name)
                        has_nulls = col.null_count > 0
                    except Exception:
                        has_nulls = False
                    nullable = field.nullable and has_nulls
                else:
                    nullable = field.nullable

        # Build type string — LowCardinality(Nullable(...)) is correct order
        if nullable:
            if base.startswith("LowCardinality("):
                inner = base[len("LowCardinality(") : -1]
                ch_type = f"LowCardinality(Nullable({inner}))"
            else:
                ch_type = f"Nullable({base})"
            # Nullable columns don't need DEFAULT
            ident = sg.to_identifier(name, quoted=True)
            return f"{ident.sql(dialect='clickhouse')} {ch_type}"
        else:
            # Non-nullable: add DEFAULT to avoid Nullable overhead (schema-types-avoid-nullable)
            # Only emit DEFAULT if original field was nullable (i.e., we are optimizing away Nullable)
            # Pure non-nullable fields could also just be type without DEFAULT, but explicit DEFAULT is clearer.
            default = None
            if field.nullable:
                # we have chosen to make it non-nullable despite Arrow saying nullable but no nulls in data
                default = _ch_default_for_base(base)
            ident = sg.to_identifier(name, quoted=True)
            if default is not None:
                return f"{ident.sql(dialect='clickhouse')} {base} DEFAULT {default}"
            return f"{ident.sql(dialect='clickhouse')} {base}"

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | pa.Table | pa.RecordBatchReader | ir.Table | None = None,
        *,
        schema: sch.SchemaLike | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
        order_by: str = "tuple()",
        partition_by: str | None = None,
        low_cardinality_cols: list[str] | set[str] | None = None,
        nullable_cols: set[str] | list[str] | bool | None = None,
        skipping_indices: list[str] | None = None,
    ) -> ir.Table:
        """Create a table, optionally populating it from ``obj``.

        ClickHouse has no auto-create on insert, so we build an explicit
        ``MergeTree`` DDL from the data/schema. Types are mapped via Arrow (the
        vendored ibis ClickHouse ``from_ibis`` mapper is incompatible with the
        pinned sqlglot), so ``obj`` or ``schema`` is converted to a pyarrow
        schema first.

        Per `schema-pk-plan-before-creation` ORDER BY is immutable — pass a real
        key (e.g. ``"user_id, event_date"``) for filter-heavy workloads. ``tuple()``
        is insertion-order with no sparse index.
        """
        import pandas as pd  # noqa: PLC0415

        if order_by == "tuple()":
            warnings.warn(
                "ORDER BY tuple() disables the sparse primary index. "
                "Pass `order_by` matching your filter columns; immutable after creation.",
                UserWarning,
                stacklevel=2,
            )

        if schema is not None:
            schema = sch.schema(schema)
        if obj is None and schema is None:
            raise ValueError("Either `obj` or `schema` must be specified")

        if obj is not None:
            if isinstance(obj, ir.Expr):
                obj = self.to_pyarrow(obj)
            elif isinstance(obj, pd.DataFrame):
                obj = pa.Table.from_pandas(obj, preserve_index=False)
            if isinstance(obj, pa.RecordBatchReader):
                obj = obj.read_all()
            arrow_schema = obj.schema
            arrow_table_for_nulls = obj
        else:
            # ibis schema -> pyarrow (working path) -> ClickHouse types
            arrow_schema = sch.schema(schema).to_pyarrow()
            arrow_table_for_nulls = None

        ident = sg.table(name, db=database, quoted=self.compiler.quoted).sql(
            dialect=self.dialect
        )
        lc_set = set(low_cardinality_cols) if low_cardinality_cols else None
        nc_val: set[str] | bool | None
        if isinstance(nullable_cols, (list, set, tuple)):
            nc_val = set(nullable_cols)
        else:
            nc_val = nullable_cols
        raw_keys = f"{order_by} {partition_by or ''}"
        key_cols = {f.name for f in arrow_schema if f.name in raw_keys}
        columns = ", ".join(
            self._arrow_field_to_clickhouse(
                f.name,
                f,
                table=arrow_table_for_nulls,
                low_cardinality_cols=lc_set,
                nullable_cols=nc_val,
                force_non_nullable=(f.name in key_cols),
            )
            for f in arrow_schema
        )
        idx_sql = ""
        if skipping_indices:
            idx_sql = ", " + ", ".join(skipping_indices)
        part_sql = f" PARTITION BY {partition_by}" if partition_by else ""
        ob = order_by.strip()
        if ob != "tuple()" and not (ob.startswith("(") and ob.endswith(")")):
            ob = f"({ob})"
        if overwrite:
            self.con.command(f"DROP TABLE IF EXISTS {ident}")
        if temp:
            warnings.warn(
                "ClickHouse TEMPORARY TABLE requested but `temp=True` is not fully wired; "
                "creating as MergeTree with ORDER BY. Use raw_sql for TEMPORARY if needed.",
                UserWarning,
                stacklevel=2,
            )
        self.con.command(
            f"CREATE TABLE {ident} ({columns}{idx_sql}) ENGINE = MergeTree{part_sql} ORDER BY {ob}"
        )
        if obj is not None:
            for start in range(0, obj.num_rows, 100_000):
                self.con.insert_arrow(name, obj.slice(start, 100_000))
        return self.table(name, database=database)

    @staticmethod
    def _from_url(url: str) -> dict[str, Any]:
        parsed = urlsplit(url)
        kwargs: dict[str, Any] = {}
        if parsed.hostname:
            kwargs["host"] = parsed.hostname
        if parsed.port:
            kwargs["port"] = parsed.port
        if parsed.username is not None:
            kwargs["username"] = parsed.username
        if parsed.password is not None:
            kwargs["password"] = parsed.password
        if parsed.path and parsed.path != "/":
            kwargs["database"] = parsed.path.lstrip("/")
        qs = parse_qs(parsed.query)
        if "secure" in qs:
            kwargs["secure"] = qs["secure"][0].lower() == "true"
        return kwargs

    @classmethod
    def connect_env(cls, **kwargs: Any):
        from xorq.common.utils.clickhouse_utils import make_connection  # noqa: PLC0415

        return make_connection(**kwargs)


def connect(*args: Any, **kwargs: Any) -> Backend:
    con = Backend()
    return con.connect(*args, **kwargs)
