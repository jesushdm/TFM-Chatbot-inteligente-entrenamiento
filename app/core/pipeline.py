"""
Pipeline de extremo a extremo: pregunta en lenguaje natural -> intencion (TF-IDF +
regresion logistica) -> entidades (reglas + RapidFuzz) -> forma logica validada ->
SQL parametrizada sobre SQLite de solo lectura -> respuesta en lenguaje natural.

Es el objeto que usan tanto la API (api/main.py) como el CLI y el script de
evaluacion (evaluate/evaluar.py), para no duplicar la logica de ensamblado.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import joblib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.db import conectar_solo_lectura  # noqa: E402
from core.entidades import UMBRAL_ACLARAR, Entidades, ResolutorEntidades  # noqa: E402
from core.forma_logica import Ambito, ErrorFormaLogica, FormaLogica  # noqa: E402
from core.responder import responder  # noqa: E402
from core.sql_templates import ejecutar  # noqa: E402

MODELO_PATH = ROOT / "nlu" / "modelo_intencion.joblib"


@dataclass
class RespuestaAsistente:
    pregunta: str
    intencion_detectada: str
    entidades: Entidades
    forma_logica: FormaLogica
    sql: str
    respuesta: str


class Asistente:
    def __init__(self):
        self.con = conectar_solo_lectura()
        self.resolutor = ResolutorEntidades(self.con)
        self.clasificador = joblib.load(MODELO_PATH)

    def cerrar(self):
        self.con.close()

    def preguntar(self, texto: str) -> RespuestaAsistente:
        intencion = self.clasificador.predict([texto])[0]
        entidades = self.resolutor.resolver(texto)
        fl = self._construir_forma_logica(intencion, entidades)
        try:
            fl.validar()
            res = ejecutar(fl, self.con)
        except ErrorFormaLogica:
            fl = FormaLogica("ambigua")
            res = ejecutar(fl, self.con)
        respuesta = responder(fl, res, detalle_ambiguedad=texto)
        return RespuestaAsistente(texto, intencion, entidades, fl, res.sql, respuesta)

    # ------------------------------------------------------------------
    def _construir_forma_logica(self, intencion: str, ent: Entidades) -> FormaLogica:
        if intencion == "fuera_de_dominio":
            return FormaLogica("fuera_de_dominio")
        if intencion == "ambigua":
            return FormaLogica("ambigua")

        s = self.resolutor.semana_actual
        ejercicio_dudoso = (
            ent.ejercicio is not None
            and ent.ejercicio.nombre_canonico is None
            and ent.ejercicio.score >= UMBRAL_ACLARAR
        )

        ambito = Ambito(
            ejercicio=ent.ejercicio.nombre_canonico if ent.ejercicio else None,
            grupo_muscular=ent.grupo_muscular,
            dia_semana=ent.dia_semana,
            semana=ent.semana,
            semana_inicio=ent.semana_inicio,
            semana_fin=ent.semana_fin,
        )

        if intencion == "consulta_registro":
            if not ambito.ejercicio:
                return FormaLogica("ambigua")
            if ambito.semana is None and ambito.semana_inicio is None:
                ambito.semana = s
            return FormaLogica("consulta_registro", ambito=ambito)

        if intencion == "consulta_maximo":
            if ambito.ejercicio:
                return FormaLogica("consulta_maximo", metrica="peso",
                                    agregacion=ent.agregacion or "maximo", ambito=ambito)
            if ejercicio_dudoso:
                return FormaLogica("ambigua")
            metrica = ent.metrica if ent.metrica in ("volumen", "rpe", "tonelaje") else "volumen"
            return FormaLogica("consulta_maximo", metrica=metrica, agregacion=ent.agregacion or "maximo",
                                ambito=Ambito(semana_inicio=ambito.semana_inicio, semana_fin=ambito.semana_fin))

        if intencion == "consulta_volumen":
            if ambito.semana_inicio is None:
                ambito.semana_inicio, ambito.semana_fin = 1, s
            return FormaLogica("consulta_volumen", ambito=ambito)

        if intencion == "consulta_frecuencia":
            if ambito.semana_inicio is None:
                ambito.semana_inicio, ambito.semana_fin = 1, s
            return FormaLogica("consulta_frecuencia", ambito=ambito)

        if intencion == "consulta_progresion":
            if ejercicio_dudoso:
                return FormaLogica("ambigua")
            top_n = None if ambito.ejercicio else 5
            if ambito.semana_inicio is None:
                ambito.semana_inicio, ambito.semana_fin = 1, s
            return FormaLogica("consulta_progresion", metrica="peso", ambito=ambito, top_n=top_n)

        if intencion == "comparacion":
            if ejercicio_dudoso:
                return FormaLogica("ambigua")
            primera = ambito.semana if ambito.semana is not None else 1
            segunda = s
            a = Ambito(ejercicio=ambito.ejercicio, grupo_muscular=ambito.grupo_muscular, semana=primera)
            b = Ambito(ejercicio=ambito.ejercicio, grupo_muscular=ambito.grupo_muscular, semana=segunda)
            return FormaLogica("comparacion", ambito=a, comparar_con=b)

        if intencion == "ultima_sesion":
            return FormaLogica("ultima_sesion", ordinal=ent.ordinal or "ultima", ambito=Ambito())

        if intencion == "consulta_catalogo":
            return FormaLogica("consulta_catalogo", ambito=Ambito(grupo_muscular=ambito.grupo_muscular))

        if intencion == "consulta_programacion":
            metrica = ent.metrica if ent.metrica in ("series", "repeticiones", "grupo_muscular", "ejercicios_distintos") else None
            if ambito.ejercicio and metrica is None:
                metrica = "repeticiones"
            if ambito.dia_semana and metrica == "series" and ambito.ejercicio is None:
                pass  # ya soportado (series por dia)
            return FormaLogica("consulta_programacion", metrica=metrica,
                                ambito=Ambito(dia_semana=ambito.dia_semana, ejercicio=ambito.ejercicio))

        return FormaLogica("ambigua")


if __name__ == "__main__":
    asistente = Asistente()
    ejemplos = [
        "¿Cuánto he progresado en press banca durante las últimas 4 semanas?",
        "¿Qué ejercicios han aumentado más de peso?",
        "¿Cuál ha sido mi mayor peso registrado en sentadilla?",
        "¿Qué ejercicios han mantenido el mismo peso durante las últimas 4 semanas?",
        "¿Cuál ha sido mi evolución del volumen de entrenamiento?",
        "¿Qué ejercicios tengo programados para el lunes?",
        "¿Cuántas series hago los martes?",
        "¿Cuántas repeticiones tengo programadas para press banca?",
        "¿Cuánto volumen semanal tengo programado?",
        "¿Qué grupos musculares entreno cada día?",
        "¿Cuántos ejercicios diferentes hago durante la semana?",
        "¿Qué peso utilicé en press banca en cada una de las últimas 4 semanas?",
        "¿Cuántas series de sentadilla hice cada semana?",
        "¿Cuánto ha aumentado el peso del remo desde la primera semana?",
        "¿Qué semana tuvo mayor volumen de entrenamiento?",
        "¿Cuál fue la semana con menor intensidad?",
        "¿Cómo ha evolucionado mi press banca durante el último mes?",
        "¿Qué ejercicios muestran una progresión de fuerza más clara?",
        "¿Cuál ha sido mi semana de entrenamiento más exigente?",
        "¿Cómo ha cambiado mi intensidad a lo largo de las 4 semanas?",
        "¿Qué diferencias hay entre mi entrenamiento de la primera y la cuarta semana?",
        "¿Qué ejercicios de pecho he realizado y cómo ha evolucionado el peso utilizado?",
        "¿Qué debería comer después de entrenar?",
        "¿Cuánto levanté en press?",
    ]
    for texto in ejemplos:
        r = asistente.preguntar(texto)
        print(f"P: {texto}")
        print(f"   intención: {r.intencion_detectada}")
        print(f"   R: {r.respuesta}")
        print()
    asistente.cerrar()
