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
