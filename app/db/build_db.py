"""
Construye db/tfm.db desde cero: aplica schema.sql e inserta los datos generados por
data/generar_datos.py (usuario sintetico, catalogo de ejercicios, plan semanal y
3 meses de historico).

Uso:
    python3 db/build_db.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.generar_datos import generar, DIAS_ENTRENO, FECHA_INICIO_BLOQUE, FECHA_REFERENCIA  # noqa: E402

DB_PATH = ROOT / "db" / "tfm.db"
SCHEMA_PATH = ROOT / "db" / "schema.sql"

# Datos del usuario sintetico (no proporcionados por el usuario; valores de relleno
# razonables, sin significado clinico ni personal real).
USUARIO_SINTETICO = {
    "nombre": "Usuario sintetico 1",
    "fecha_nacimiento": "1996-03-10",
    "sexo": "M",
    "altura_cm": 178.0,
    "nivel": "Principiante-intermedio",
}


def construir(db_path: Path = DB_PATH) -> None:
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON;")
    con.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    datos = generar()

    cur = con.cursor()

    cur.execute(
        "INSERT INTO usuario (nombre, fecha_nacimiento, sexo, altura_cm, nivel) "
        "VALUES (:nombre, :fecha_nacimiento, :sexo, :altura_cm, :nivel)",
        USUARIO_SINTETICO,
    )
    usuario_id = cur.lastrowid

    ejercicio_id = {}
    for e in datos.ejercicios:
        cur.execute(
            "INSERT INTO ejercicio (nombre, grupo_muscular, equipamiento, unilateral, es_isometrico) "
            "VALUES (?, ?, ?, ?, ?)",
            (e["nombre"], e["grupo_muscular"], e["equipamiento"], e["unilateral"], e["es_isometrico"]),
        )
        ejercicio_id[e["nombre"]] = cur.lastrowid

    for nombre, alias in datos.alias:
        cur.execute(
            "INSERT INTO alias_ejercicio (ejercicio_id, alias) VALUES (?, ?)",
            (ejercicio_id[nombre], alias),
        )

    cur.execute(
        "INSERT INTO rutina (nombre, fecha_inicio, fecha_fin, objetivo) VALUES (?, ?, ?, ?)",
        (
            "Fuerza 3 dias - Torso/Pierna/Torso",
            FECHA_INICIO_BLOQUE.isoformat(),
            FECHA_REFERENCIA.isoformat(),
            "Fuerza e hipertrofia",
        ),
    )
    rutina_id = cur.lastrowid

    orden_por_dia = {d: 0 for d in DIAS_ENTRENO}
    for fp in datos.plan:
        orden_por_dia[fp.dia] += 1
        cur.execute(
            "INSERT INTO rutina_ejercicio "
            "(rutina_id, dia_semana, ejercicio_id, orden, series_objetivo, rep_min, rep_max, rir_objetivo) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (rutina_id, fp.dia, ejercicio_id[fp.ejercicio], orden_por_dia[fp.dia],
             fp.series, fp.rep_min, fp.rep_max, fp.rir),
        )

    sesion_id = {}
    for s in datos.sesiones:
        cur.execute(
            "INSERT INTO sesion (usuario_id, rutina_id, fecha, semana, dia_semana, duracion_min, rpe_sesion) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (usuario_id, rutina_id, s["fecha"], s["semana"], s["dia"], s["duracion_min"], s["rpe_sesion"]),
        )
        sesion_id[(s["semana"], s["dia"])] = cur.lastrowid

    n_series = 0
    for (semana, dia), filas in datos.series_por_sesion.items():
        sid = sesion_id[(semana, dia)]
        for orden, f in enumerate(filas, start=1):
            for numero_serie in range(1, f.series + 1):
                cur.execute(
                    "INSERT INTO serie "
                    "(sesion_id, ejercicio_id, orden_ejercicio, numero_serie, repeticiones, peso_kg, rir, tipo_serie) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 'efectiva')",
                    (sid, ejercicio_id[f.ejercicio], orden, numero_serie, f.repeticiones, f.peso_kg, f.rir),
                )
                n_series += 1

    con.commit()
    con.close()

    print(f"Base de datos creada en {db_path}")
    print(f"  usuario: 1 | ejercicios: {len(ejercicio_id)} | plan (rutina_ejercicio): {len(datos.plan)}")
    print(f"  sesiones: {len(datos.sesiones)} | filas de serie (con desglose por numero_serie): {n_series}")


if __name__ == "__main__":
    construir()
