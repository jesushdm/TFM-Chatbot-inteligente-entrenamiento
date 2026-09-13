"""Conexion de solo lectura a la base de datos SQLite del asistente.

Se abre en modo 'ro' (read-only) via URI para que la capa de acceso a datos no pueda
modificar el historico bajo ninguna circunstancia, tal como se describe en el
apartado 3.7 de la Memoria (Cap.3).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "db" / "tfm.db"


def conectar_solo_lectura(db_path: Path = DB_PATH) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    return con
