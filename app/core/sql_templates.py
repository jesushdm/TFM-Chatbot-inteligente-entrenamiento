"""
Plantillas SQL parametrizadas: traducen una FormaLogica validada en una consulta
SQL de solo lectura sobre tfm.db. Es la alternativa elegida en el Cap.3 frente a
RAG y a texto-a-SQL generativo (ver comparativa en la Memoria, apartado 3.7).

Cada funcion publica devuelve un objeto Resultado con el SQL ejecutado (para poder
usarlo como "consulta SQL de referencia" en el dataset anotado), los parametros y
las filas obtenidas.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from core.forma_logica import Ambito, ErrorFormaLogica, FormaLogica


@dataclass
class Resultado:
    sql: str
    params: tuple
    filas: list[dict] = field(default_factory=list)


def _filas_a_dicts(cur: sqlite3.Cursor) -> list[dict]:
    return [dict(r) for r in cur.fetchall()]


def _clausulas_ambito(amb: Ambito, alias_serie: str = "se", alias_sesion: str = "s", alias_ejercicio: str = "e"):
    """Construye fragmentos WHERE reutilizables para consultas sobre serie/sesion."""
    condiciones = []
    params: list = []
    if amb.ejercicio:
        condiciones.append(f"{alias_ejercicio}.nombre = ?")
        params.append(amb.ejercicio)
    if amb.grupo_muscular:
        condiciones.append(f"{alias_ejercicio}.grupo_muscular = ?")
        params.append(amb.grupo_muscular)
    if amb.dia_semana:
        condiciones.append(f"{alias_sesion}.dia_semana = ?")
        params.append(amb.dia_semana)
    if amb.semana is not None:
        condiciones.append(f"{alias_sesion}.semana = ?")
        params.append(amb.semana)
    if amb.semana_inicio is not None and amb.semana_fin is not None:
        condiciones.append(f"{alias_sesion}.semana BETWEEN ? AND ?")
        params.extend([amb.semana_inicio, amb.semana_fin])
    return condiciones, params


def consulta_registro(fl: FormaLogica, con: sqlite3.Connection) -> Resultado:
    amb = fl.ambito
    condiciones, params = _clausulas_ambito(amb)
    where = ("WHERE " + " AND ".join(condiciones)) if condiciones else ""
    sql = (
        "SELECT s.semana, s.dia_semana, s.fecha, e.nombre AS ejercicio, "
        "se.numero_serie, se.peso_kg, se.repeticiones, se.rir "
        "FROM serie se "
        "JOIN sesion s ON se.sesion_id = s.id "
        "JOIN ejercicio e ON se.ejercicio_id = e.id "
        f"{where} "
        "ORDER BY s.semana, s.fecha, se.numero_serie"
    )
    cur = con.execute(sql, params)
    return Resultado(sql, tuple(params), _filas_a_dicts(cur))


def consulta_maximo(fl: FormaLogica, con: sqlite3.Connection) -> Resultado:
    amb = fl.ambito
    orden = "DESC" if fl.agregacion != "minimo" else "ASC"
    metrica = fl.metrica or "peso"

    if metrica == "peso":
        condiciones, params = _clausulas_ambito(amb)
        where = ("WHERE " + " AND ".join(condiciones)) if condiciones else ""
        sql = (
            "SELECT s.semana, s.dia_semana, s.fecha, e.nombre AS ejercicio, se.peso_kg, se.repeticiones "
            "FROM serie se JOIN sesion s ON se.sesion_id = s.id JOIN ejercicio e ON se.ejercicio_id = e.id "
            f"{where} ORDER BY se.peso_kg {orden} LIMIT 1"
        )
        cur = con.execute(sql, params)
        return Resultado(sql, tuple(params), _filas_a_dicts(cur))

    # metrica en {'volumen', 'rpe', 'tonelaje'} agregada por semana (p.ej. "que semana
    # tuvo mayor volumen" / "cual fue la semana con menor intensidad")
    if metrica == "rpe":
        # La intensidad se agrega a nivel de sesion (no depende de ejercicio/grupo).
        condiciones, params = [], []
        if amb.semana_inicio is not None and amb.semana_fin is not None:
            condiciones.append("semana BETWEEN ? AND ?")
            params.extend([amb.semana_inicio, amb.semana_fin])
        where = ("WHERE " + " AND ".join(condiciones)) if condiciones else ""
        sql = (
            "SELECT semana, AVG(rpe_sesion) AS valor "
            f"FROM sesion {where} "
            f"GROUP BY semana ORDER BY valor {orden} LIMIT 1"
        )
        cur = con.execute(sql, params)
        return Resultado(sql, tuple(params), _filas_a_dicts(cur))

    condiciones, params = _clausulas_ambito(amb)
    where = ("WHERE " + " AND ".join(condiciones)) if condiciones else ""
    expr = "SUM(se.peso_kg * se.repeticiones)" if metrica == "tonelaje" else "COUNT(*)"
    sql = (
        f"SELECT s.semana, {expr} AS valor "
        "FROM serie se JOIN sesion s ON se.sesion_id = s.id JOIN ejercicio e ON se.ejercicio_id = e.id "
        f"{where} GROUP BY s.semana ORDER BY valor {orden} LIMIT 1"
    )
    cur = con.execute(sql, params)
    return Resultado(sql, tuple(params), _filas_a_dicts(cur))


def consulta_volumen(fl: FormaLogica, con: sqlite3.Connection) -> Resultado:
    amb = fl.ambito
    condiciones, params = _clausulas_ambito(amb)
    where = ("WHERE " + " AND ".join(condiciones)) if condiciones else ""
    sql = (
        "SELECT s.semana, COUNT(*) AS series, SUM(se.peso_kg * se.repeticiones) AS tonelaje "
        "FROM serie se JOIN sesion s ON se.sesion_id = s.id JOIN ejercicio e ON se.ejercicio_id = e.id "
        f"{where} GROUP BY s.semana ORDER BY s.semana"
    )
    cur = con.execute(sql, params)
    return Resultado(sql, tuple(params), _filas_a_dicts(cur))


def consulta_frecuencia(fl: FormaLogica, con: sqlite3.Connection) -> Resultado:
    amb = fl.ambito
    if amb.ejercicio or amb.grupo_muscular:
        condiciones, params = _clausulas_ambito(amb)
        where = "WHERE " + " AND ".join(condiciones)
        sql = (
            "SELECT COUNT(DISTINCT s.id) AS sesiones "
            "FROM serie se JOIN sesion s ON se.sesion_id = s.id JOIN ejercicio e ON se.ejercicio_id = e.id "
            f"{where}"
        )
        cur = con.execute(sql, params)
    else:
        condiciones = []
        params: list = []
        if amb.semana_inicio is not None and amb.semana_fin is not None:
            condiciones.append("semana BETWEEN ? AND ?")
            params.extend([amb.semana_inicio, amb.semana_fin])
        if amb.semana is not None:
            condiciones.append("semana = ?")
            params.append(amb.semana)
        where = ("WHERE " + " AND ".join(condiciones)) if condiciones else ""
        sql = f"SELECT COUNT(*) AS sesiones FROM sesion {where}"
        cur = con.execute(sql, params)
    return Resultado(sql, tuple(params), _filas_a_dicts(cur))


def consulta_progresion(fl: FormaLogica, con: sqlite3.Connection) -> Resultado:
    amb = fl.ambito
    if amb.ejercicio:
        # Serie temporal semana -> peso maximo de esa semana, para un ejercicio concreto.
        condiciones, params = _clausulas_ambito(amb)
        where = "WHERE " + " AND ".join(condiciones)
        sql = (
            "SELECT s.semana, MAX(se.peso_kg) AS peso_kg "
            "FROM serie se JOIN sesion s ON se.sesion_id = s.id JOIN ejercicio e ON se.ejercicio_id = e.id "
            f"{where} GROUP BY s.semana ORDER BY s.semana"
        )
        cur = con.execute(sql, params)
        return Resultado(sql, tuple(params), _filas_a_dicts(cur))

    # Sin ejercicio: ranking de todos los ejercicios por variacion de peso entre la
    # primera y la ultima semana con datos (p.ej. "que ejercicios han aumentado mas
    # de peso" / "que ejercicios han mantenido el mismo peso").
    condiciones, params = [], []
    if amb.semana_inicio is not None and amb.semana_fin is not None:
        condiciones.append("s.semana BETWEEN ? AND ?")
        params.extend([amb.semana_inicio, amb.semana_fin])
    where = ("AND " + " AND ".join(condiciones)) if condiciones else ""
    sql = (
        "WITH por_semana AS ("
        "  SELECT e.nombre AS ejercicio, s.semana, MAX(se.peso_kg) AS peso_kg "
        "  FROM serie se JOIN sesion s ON se.sesion_id = s.id JOIN ejercicio e ON se.ejercicio_id = e.id "
        f"  WHERE se.peso_kg IS NOT NULL {where} "
        "  GROUP BY e.nombre, s.semana"
        "), extremos AS ("
        "  SELECT ejercicio, MIN(semana) AS semana_ini, MAX(semana) AS semana_fin "
        "  FROM por_semana GROUP BY ejercicio"
        ") "
        "SELECT p1.ejercicio, p1.peso_kg AS peso_inicial, p2.peso_kg AS peso_final, "
        "       (p2.peso_kg - p1.peso_kg) AS delta "
        "FROM extremos ex "
        "JOIN por_semana p1 ON p1.ejercicio = ex.ejercicio AND p1.semana = ex.semana_ini "
        "JOIN por_semana p2 ON p2.ejercicio = ex.ejercicio AND p2.semana = ex.semana_fin "
        "ORDER BY delta DESC"
    )
    cur = con.execute(sql, params)
    return Resultado(sql, tuple(params), _filas_a_dicts(cur))


def comparacion(fl: FormaLogica, con: sqlite3.Connection) -> Resultado:
    if fl.comparar_con is None:
        raise ErrorFormaLogica("comparacion requiere comparar_con")

    def _agregado(amb: Ambito):
        condiciones, params = _clausulas_ambito(amb)
        where = ("WHERE " + " AND ".join(condiciones)) if condiciones else ""
        sql = (
            "SELECT COUNT(*) AS series, SUM(se.peso_kg * se.repeticiones) AS tonelaje, "
            "MAX(se.peso_kg) AS peso_max "
            "FROM serie se JOIN sesion s ON se.sesion_id = s.id JOIN ejercicio e ON se.ejercicio_id = e.id "
            f"{where}"
        )
        cur = con.execute(sql, params)
        return sql, params, _filas_a_dicts(cur)

    sql_a, params_a, filas_a = _agregado(fl.ambito)
    sql_b, params_b, filas_b = _agregado(fl.comparar_con)
    filas = [
        {"ambito": "A", **filas_a[0]},
        {"ambito": "B", **filas_b[0]},
    ]
    sql_completo = f"-- Ambito A:\n{sql_a}\n-- Ambito B:\n{sql_b}"
    return Resultado(sql_completo, (tuple(params_a), tuple(params_b)), filas)


def ultima_sesion(fl: FormaLogica, con: sqlite3.Connection) -> Resultado:
    offset = 0
    if isinstance(fl.ordinal, int):
        offset = fl.ordinal - 1
    elif fl.ordinal == "penultima":
        offset = 1
    sql_sesion = "SELECT * FROM sesion ORDER BY fecha DESC LIMIT 1 OFFSET ?"
    cur = con.execute(sql_sesion, (offset,))
    filas_sesion = _filas_a_dicts(cur)
    if not filas_sesion:
        return Resultado(sql_sesion, (offset,), [])
    sesion_id = filas_sesion[0]["id"]
    sql_series = (
        "SELECT e.nombre AS ejercicio, se.numero_serie, se.peso_kg, se.repeticiones, se.rir "
        "FROM serie se JOIN ejercicio e ON se.ejercicio_id = e.id "
        "WHERE se.sesion_id = ? ORDER BY se.orden_ejercicio, se.numero_serie"
    )
    cur = con.execute(sql_series, (sesion_id,))
    filas_series = _filas_a_dicts(cur)
    sql = f"{sql_sesion}\n{sql_series}"
    return Resultado(sql, (offset, sesion_id), [{"sesion": filas_sesion[0], "series": filas_series}])


def consulta_catalogo(fl: FormaLogica, con: sqlite3.Connection) -> Resultado:
    amb = fl.ambito
    condiciones, params = [], []
    if amb.grupo_muscular:
        condiciones.append("grupo_muscular = ?")
        params.append(amb.grupo_muscular)
    where = ("WHERE " + " AND ".join(condiciones)) if condiciones else ""
    sql = f"SELECT nombre, grupo_muscular, equipamiento FROM ejercicio {where} ORDER BY nombre"
    cur = con.execute(sql, params)
    return Resultado(sql, tuple(params), _filas_a_dicts(cur))


def consulta_programacion(fl: FormaLogica, con: sqlite3.Connection) -> Resultado:
    amb = fl.ambito
    metrica = fl.metrica

    if metrica == "series" and not amb.ejercicio:
        condiciones, params = [], []
        if amb.dia_semana:
            condiciones.append("dia_semana = ?")
            params.append(amb.dia_semana)
        where = ("WHERE " + " AND ".join(condiciones)) if condiciones else ""
        sql = f"SELECT dia_semana, SUM(series_objetivo) AS series FROM rutina_ejercicio {where} GROUP BY dia_semana"
        cur = con.execute(sql, params)
        return Resultado(sql, tuple(params), _filas_a_dicts(cur))

    if metrica == "repeticiones" and amb.ejercicio:
        sql = (
            "SELECT e.nombre AS ejercicio, re.dia_semana, re.series_objetivo, re.rep_min, re.rep_max, re.rir_objetivo "
            "FROM rutina_ejercicio re JOIN ejercicio e ON re.ejercicio_id = e.id "
            "WHERE e.nombre = ?"
        )
        cur = con.execute(sql, (amb.ejercicio,))
        return Resultado(sql, (amb.ejercicio,), _filas_a_dicts(cur))

    if metrica == "grupo_muscular":
        sql = (
            "SELECT DISTINCT re.dia_semana, e.grupo_muscular "
            "FROM rutina_ejercicio re JOIN ejercicio e ON re.ejercicio_id = e.id "
            "ORDER BY re.dia_semana"
        )
        cur = con.execute(sql)
        return Resultado(sql, (), _filas_a_dicts(cur))

    if metrica == "ejercicios_distintos":
        sql = "SELECT COUNT(DISTINCT ejercicio_id) AS ejercicios FROM rutina_ejercicio"
        cur = con.execute(sql)
        return Resultado(sql, (), _filas_a_dicts(cur))

    # Caso por defecto: listar ejercicios programados (de un dia o de toda la semana)
    condiciones, params = [], []
    if amb.dia_semana:
        condiciones.append("re.dia_semana = ?")
        params.append(amb.dia_semana)
    if amb.ejercicio:
        condiciones.append("e.nombre = ?")
        params.append(amb.ejercicio)
    where = ("WHERE " + " AND ".join(condiciones)) if condiciones else ""
    sql = (
        "SELECT re.dia_semana, e.nombre AS ejercicio, e.grupo_muscular, "
        "re.series_objetivo, re.rep_min, re.rep_max, re.rir_objetivo "
        "FROM rutina_ejercicio re JOIN ejercicio e ON re.ejercicio_id = e.id "
        f"{where} ORDER BY re.dia_semana, re.orden"
    )
    cur = con.execute(sql, params)
    return Resultado(sql, tuple(params), _filas_a_dicts(cur))


DESPACHADOR = {
    "consulta_registro": consulta_registro,
    "consulta_maximo": consulta_maximo,
    "consulta_volumen": consulta_volumen,
    "consulta_frecuencia": consulta_frecuencia,
    "consulta_progresion": consulta_progresion,
    "comparacion": comparacion,
    "ultima_sesion": ultima_sesion,
    "consulta_catalogo": consulta_catalogo,
    "consulta_programacion": consulta_programacion,
}


def ejecutar(fl: FormaLogica, con: sqlite3.Connection) -> Resultado:
    fl.validar()
    if fl.intencion in ("fuera_de_dominio", "ambigua"):
        return Resultado(sql="", params=(), filas=[])
    funcion = DESPACHADOR.get(fl.intencion)
    if funcion is None:
        raise ErrorFormaLogica(f"Sin plantilla SQL para la intencion: {fl.intencion}")
    return funcion(fl, con)
