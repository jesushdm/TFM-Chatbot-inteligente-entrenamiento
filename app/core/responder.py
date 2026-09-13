"""
Generador de respuesta en lenguaje natural a partir del resultado de ejecutar una
FormaLogica (ver apartado 3.7 de la Memoria: capa de respuesta con manejo de
"sin datos", aclaracion y rechazo fuera de dominio).
"""
from __future__ import annotations

from core.forma_logica import FormaLogica
from core.sql_templates import Resultado

MENSAJE_FUERA_DE_DOMINIO = (
    "Esa pregunta no se puede responder con los datos de entrenamiento del sistema "
    "(rutina programada e historico de sesiones). Prueba a preguntar sobre tus "
    "ejercicios, series, pesos o el plan semanal."
)
MENSAJE_AMBIGUA_PLANTILLA = (
    "No tengo claro a qué te refieres con \"{detalle}\". ¿Podrías indicar el ejercicio "
    "o el periodo concreto (p.ej. \"press banca\", \"las últimas 4 semanas\")?"
)
MENSAJE_SIN_DATOS = "No encuentro datos para esa combinación de ejercicio y periodo."


def _fmt(n) -> str:
    if n is None:
        return "sin dato"
    if isinstance(n, float):
        return f"{n:.1f}".rstrip("0").rstrip(".")
    return str(n)


def responder(fl: FormaLogica, res: Resultado, detalle_ambiguedad: str | None = None) -> str:
    if fl.intencion == "fuera_de_dominio":
        return MENSAJE_FUERA_DE_DOMINIO
    if fl.intencion == "ambigua":
        return MENSAJE_AMBIGUA_PLANTILLA.format(detalle=detalle_ambiguedad or "tu pregunta")

    if not res.filas:
        return MENSAJE_SIN_DATOS

    if fl.intencion == "consulta_registro":
        partes = [
            f"semana {f['semana']} ({f['dia_semana']}): {_fmt(f['peso_kg'])} kg x {f['repeticiones']} "
            f"(RIR {f['rir']})" for f in res.filas
        ]
        ejercicio = fl.ambito.ejercicio or "ese ejercicio"
        return f"Registro de {ejercicio} — " + "; ".join(partes) + "."

    if fl.intencion == "consulta_maximo":
        f = res.filas[0]
        superlativo = "menor" if fl.agregacion == "minimo" else "mayor"
        if "peso_kg" in f:
            return (
                f"Tu {superlativo} peso registrado en {fl.ambito.ejercicio} es de "
                f"{_fmt(f['peso_kg'])} kg ({_fmt(f['repeticiones'])} repeticiones, semana {f['semana']})."
            )
        return f"La semana con {superlativo} valor ({fl.metrica}) fue la semana {f['semana']}, con {_fmt(f['valor'])}."

    if fl.intencion == "consulta_volumen":
        total_series = sum(f["series"] for f in res.filas)
        detalle = ", ".join(f"semana {f['semana']}: {f['series']} series" for f in res.filas)
        return f"Volumen total: {total_series} series. Desglose por semana — {detalle}."

    if fl.intencion == "consulta_frecuencia":
        n = res.filas[0].get("sesiones", 0)
        return f"Has entrenado {n} sesiones en ese periodo."

    if fl.intencion == "consulta_progresion":
        if fl.ambito.ejercicio:
            primero, ultimo = res.filas[0], res.filas[-1]
            delta = (ultimo["peso_kg"] or 0) - (primero["peso_kg"] or 0)
            tendencia = "aumentado" if delta > 0 else ("mantenido" if delta == 0 else "reducido")
            serie_txt = ", ".join(f"sem.{f['semana']}: {_fmt(f['peso_kg'])} kg" for f in res.filas)
            return (
                f"Has {tendencia} {_fmt(abs(delta))} kg en {fl.ambito.ejercicio} "
                f"(de {_fmt(primero['peso_kg'])} a {_fmt(ultimo['peso_kg'])} kg). Evolución: {serie_txt}."
            )
        top = res.filas[: fl.top_n or 3]
        detalle = ", ".join(f"{f['ejercicio']} (+{_fmt(f['delta'])} kg)" for f in top if f["delta"] and f["delta"] > 0)
        return f"Los ejercicios que más han aumentado de peso son: {detalle or 'ninguno en ese periodo'}."

    if fl.intencion == "comparacion":
        a, b = res.filas[0], res.filas[1]
        return (
            f"Ámbito A: {a['series']} series, {_fmt(a['tonelaje'])} kg de tonelaje, "
            f"máximo {_fmt(a['peso_max'])} kg. Ámbito B: {b['series']} series, "
            f"{_fmt(b['tonelaje'])} kg de tonelaje, máximo {_fmt(b['peso_max'])} kg. "
            f"Diferencia de tonelaje: {_fmt((b['tonelaje'] or 0) - (a['tonelaje'] or 0))} kg."
        )

    if fl.intencion == "ultima_sesion":
        info = res.filas[0]
        s = info["sesion"]
        series = info["series"]
        ejercicios = sorted({f["ejercicio"] for f in series})
        return (
            f"Tu sesión más reciente fue el {s['fecha']} ({s['dia_semana']}, semana {s['semana']}), "
            f"con {len(series)} series de {len(ejercicios)} ejercicios: {', '.join(ejercicios)}."
        )

    if fl.intencion == "consulta_catalogo":
        nombres = [f["nombre"] for f in res.filas]
        return f"Ejercicios registrados: {', '.join(nombres)}." if nombres else MENSAJE_SIN_DATOS

    if fl.intencion == "consulta_programacion":
        metrica = fl.metrica
        if metrica == "series" and "series" in res.filas[0] and "dia_semana" in res.filas[0]:
            detalle = ", ".join(f"{f['dia_semana']}: {f['series']} series" for f in res.filas)
            return f"Series programadas — {detalle}."
        if metrica == "repeticiones":
            f = res.filas[0]
            return (
                f"{f['ejercicio']} está programado el {f['dia_semana']}: {f['series_objetivo']} series de "
                f"{f['rep_min']}-{f['rep_max']} repeticiones (RIR objetivo {f['rir_objetivo']})."
            )
        if metrica == "grupo_muscular":
            por_dia: dict[str, list[str]] = {}
            for f in res.filas:
                por_dia.setdefault(f["dia_semana"], []).append(f["grupo_muscular"])
            detalle = "; ".join(f"{d}: {', '.join(gs)}" for d, gs in por_dia.items())
            return f"Grupos musculares programados por día — {detalle}."
        if metrica == "ejercicios_distintos":
            return f"Tienes {res.filas[0]['ejercicios']} ejercicios distintos programados durante la semana."
        # listado por defecto
        detalle = "; ".join(
            f"{f['ejercicio']} ({f['series_objetivo']}x{f['rep_min']}-{f['rep_max']}, RIR {f['rir_objetivo']})"
            for f in res.filas
        )
        dia = fl.ambito.dia_semana or "la semana"
        return f"Programado para {dia}: {detalle}."

    return MENSAJE_SIN_DATOS
