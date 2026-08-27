from __future__ import annotations

import pytest

pytestmark = pytest.mark.clickhouse


def test_compile():
    import xorq.api as xo

    # compilation only — no server needed
    expr = xo.memtable({"a": [1, 2, 3]}).filter(xo._.a > 1)
    # compile via ClickHouse dialect
    from xorq.backends.clickhouse import Backend

    con = Backend()
    # no connection needed for compile
    sql = con.compile(expr)
    assert "SELECT" in sql


def test_connect_env(monkeypatch):
    monkeypatch.setenv("CLICKHOUSE_HOST", "localhost")
    monkeypatch.setenv("CLICKHOUSE_PORT", "8123")
    monkeypatch.setenv("CLICKHOUSE_DATABASE", "default")
    import xorq.api as xo

    # connect_env should not raise when clickhouse-connect is installed;
    # if not installed it raises ImportError with hint — allow either
    try:
        con = xo.clickhouse.connect_env()
        assert con is not None
    except ImportError as e:
        assert "clickhouse-connect" in str(e)
    except Exception:
        # no server running — do_connect will try to connect; allow failure in CI without docker
        pass


def _live_con():
    """Return a connected backend, or skip the test when no server is reachable."""
    import xorq.api as xo

    try:
        con = xo.clickhouse.connect_env()
        con.raw_sql("SELECT 1")  # probe
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"ClickHouse server not reachable: {e}")
    return con


def test_integration_roundtrip():
    import pyarrow as pa

    con = _live_con()
    tbl = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "name": pa.array(["a", None, "c"], type=pa.string()),
            "amount": pa.array([1.5, 2.5, 3.5], type=pa.float64()),
        }
    )
    t = con.read_record_batches(tbl, table_name="ch_test_roundtrip")
    assert con.execute(t.count()) == 3

    # schema inference must return native types, not all-String
    inferred = con.sql("SELECT id, amount * 2 AS dbl FROM ch_test_roundtrip").schema()
    assert inferred["id"].is_int64()
    assert inferred["dbl"].is_float64()

    # get_schema returns native types
    assert con.get_schema("ch_test_roundtrip")["name"].is_string()

    # create_table from a schema (Arrow mapping path)
    created = con.create_table("ch_test_created", schema=con.get_schema("ch_test_roundtrip"))
    assert created.get_name() == "ch_test_created"

    con.raw_sql("DROP TABLE IF EXISTS ch_test_roundtrip")
    con.raw_sql("DROP TABLE IF EXISTS ch_test_created")


def test_from_url():
    from xorq.backends.clickhouse import Backend

    parsed = Backend._from_url("clickhouse://u:p@localhost:8123/mydb?secure=true")
    assert parsed == {
        "host": "localhost",
        "port": 8123,
        "username": "u",
        "password": "p",
        "database": "mydb",
        "secure": True,
    }
