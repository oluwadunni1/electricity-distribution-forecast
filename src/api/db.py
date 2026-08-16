"""
db.py

Logs one row per successful prediction to the `predictions` table
(see sql/001_create_predictions_table.sql). Uses a small synchronous
connection pool opened once at API startup.

Note: a sync pool called from an async endpoint blocks the event loop
briefly per request — a deliberate simplification for demo-scale traffic.
At real production volume, swap ConnectionPool for psycopg_pool's
AsyncConnectionPool instead; the SQL and table schema don't need to change,
only how the connection is awaited.
"""
import os
from datetime import datetime

from psycopg_pool import ConnectionPool
from psycopg.types.json import Jsonb

_INSERT_SQL = """
INSERT INTO predictions
    (target_timestamp, features, predicted_load_mw, model_name, model_version)
VALUES
    (%(target_timestamp)s, %(features)s, %(predicted_load_mw)s, %(model_name)s, %(model_version)s)
ON CONFLICT (target_timestamp, model_version) DO NOTHING
"""


def create_pool() -> ConnectionPool:
    return ConnectionPool(conninfo=os.environ["DATABASE_URL"], min_size=1, max_size=5, open=True)


def log_prediction(
    pool: ConnectionPool,
    target_timestamp: datetime,
    features: dict,
    predicted_load_mw: float,
    model_name: str,
    model_version: str,
) -> None:
    with pool.connection() as conn:
        conn.execute(
            _INSERT_SQL,
            {
                "target_timestamp": target_timestamp,
                "features": Jsonb(features),
                "predicted_load_mw": predicted_load_mw,
                "model_name": model_name,
                "model_version": model_version,
            },
        )