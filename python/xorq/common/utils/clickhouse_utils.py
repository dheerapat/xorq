from __future__ import annotations

from attr import field, frozen
from attr.validators import instance_of

from xorq.backends.clickhouse import Backend as ClickHouseBackend
from xorq.common.utils.env_utils import EnvConfigable, env_templates_dir


ClickHouseConfig = EnvConfigable.subclass_from_env_file(
    env_templates_dir.joinpath(".env.clickhouse.template")
)
clickhouse_config = ClickHouseConfig.from_env()


def make_connection_params():
    return {
        "host": clickhouse_config["CLICKHOUSE_HOST"] or "localhost",
        "port": int(clickhouse_config["CLICKHOUSE_PORT"] or 8123),
        "database": clickhouse_config["CLICKHOUSE_DATABASE"] or "default",
        "username": clickhouse_config["CLICKHOUSE_USER"] or "default",
        "password": clickhouse_config["CLICKHOUSE_PASSWORD"] or "",
        "secure": (clickhouse_config["CLICKHOUSE_SECURE"] or "false").lower() == "true",
    }


def make_connection(**kwargs):
    con = ClickHouseBackend()
    params = {**make_connection_params(), **kwargs}
    # coerce port if passed as str via env
    if "port" in params and isinstance(params["port"], str):
        params["port"] = int(params["port"])
    return con.connect(**params)


@frozen
class ClickHouseADBC:
    con = field(validator=instance_of(ClickHouseBackend))
    ingest_install_hint = 'pip install "xorq[clickhouse]"'

    def get_conn(self, **kwargs):
        # clickhouse-connect is not DB-API/ADBC; use raw client for ingest
        return self.con.con
