"""
Resolucion de entidades (slot filling) por reglas y diccionarios, con fuzzy matching
(RapidFuzz) contra alias_ejercicio para el nombre del ejercicio. Es el enfoque elegido
para esta fase de la implementacion (ver apartado 3.6 de la Memoria y la nota de
estado añadida sobre el motor NLU): mas simple que entrenar una cabeza de etiquetado
BIO con un encoder, y suficiente para el alcance actual del asistente.

RapidFuzz se usa exactamente para el papel que ya describia el Cap.3 (resolucion de
la distancia entre el vocabulario del usuario y el nombre canonico del ejercicio).
"""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

UMBRAL_ACEPTAR = 80
UMBRAL_ACLARAR = 68

GRUPOS_SINONIMOS = {
    "Pecho": ["pecho", "pectoral", "pectorales"],
    "Espalda": ["espalda", "dorsal", "dorsales", "espaldas"],
    "Hombro": ["hombro", "hombros", "deltoides"],
    "Biceps": ["biceps", "bíceps"],
    "Triceps": ["triceps", "tríceps"],
    "Piernas": ["pierna", "piernas", "cuadriceps", "cuádriceps", "femoral", "gluteo", "gemelo", "gemelos"],
    "Core": ["core", "abdomen", "abdominales", "abdominal"],
}

DIAS_SINONIMOS = {
    "Lunes": ["lunes"],
    "Martes": ["martes"],
    "Miercoles": ["miercoles", "miércoles"],
}

METRICA_SINONIMOS = {
    "peso": ["peso", "carga", "kilos", "kg"],
    "series": ["series", "series efectivas"],
    "repeticiones": ["repeticiones", "reps"],
    "volumen": ["volumen"],
    "tonelaje": ["tonelaje"],
    "rpe": ["intensidad", "esfuerzo", "rpe"],
    "duracion": ["duracion", "duración", "duracion de la sesion"],
}

AGREGACION_PATRONES = [
    (r"\b(mayor|máximo|maximo|mejor marca|más pesado|record|récord)\b", "maximo"),
    (r"\b(menor|mínimo|minimo|más ligero|peor)\b", "minimo"),
    (r"\b(media|promedio)\b", "media"),
    (r"\b(total|suma|acumulad[oa])\b", "total"),
]

ORDINAL_PATRONES = [
    (r"\b(ultima|última)\b", "ultima"),
    (r"\bpenultima|penúltima\b", "penultima"),
]


STOPWORDS = {
    "el", "la", "los", "las", "de", "del", "en", "que", "para", "cada", "dia", "dias",
    "mi", "he", "ha", "un", "una", "y", "o", "a", "al", "es", "fue", "ese", "esa",
    "cual", "cuanto", "cuanta", "cuantos", "cuantas", "que", "con", "durante", "mas",
    "tengo", "hago", "hice", "se", "su", "sus", "lo", "le", "he",
    # "peso"/"pesos" se excluyen deliberadamente de las ventanas de fuzzy matching:
    # es una palabra generica muy frecuente en las preguntas ("que peso utilice...")
    # que por pura coincidencia de subcadena puntuaba alto contra el alias "peso
    # muerto", generando falsos positivos de ejercicio en preguntas que no mencionan
    # ningun ejercicio. La coincidencia EXACTA de "peso muerto" como alias completo
    # (paso 1, subcadena literal) no se ve afectada por esta exclusion.
    "peso", "pesos",
}


def _norm(txt: str) -> str:
    txt = txt.lower().strip()
    txt = "".join(c for c in unicodedata.normalize("NFD", txt) if unicodedata.category(c) != "Mn")
    txt = re.sub(r"[^\w\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


@dataclass
class EntidadEjercicio:
    nombre_canonico: str | None
    score: float
    candidato_texto: str | None = None


@dataclass
class Entidades:
    ejercicio: EntidadEjercicio | None = None
    grupo_muscular: str | None = None
    dia_semana: str | None = None
    metrica: str | None = None
    agregacion: str | None = None
    ordinal: str | None = None
    semana: int | None = None
    semana_inicio: int | None = None
    semana_fin: int | None = None
    umbral: float | None = None
    umbral_operador: str | None = None


class ResolutorEntidades:
    """Carga el catalogo de ejercicios/alias una vez y resuelve entidades sobre texto."""

    def __init__(self, con: sqlite3.Connection):
        self._alias_a_canonico: dict[str, str] = {}
        cur = con.execute("SELECT nombre FROM ejercicio")
        for (nombre,) in cur.fetchall():
            self._alias_a_canonico[_norm(nombre)] = nombre
        cur = con.execute(
            "SELECT e.nombre, a.alias FROM alias_ejercicio a JOIN ejercicio e ON a.ejercicio_id = e.id"
        )
        for nombre, alias in cur.fetchall():
            self._alias_a_canonico[_norm(alias)] = nombre
        cur = con.execute("SELECT MAX(semana) FROM sesion")
        self.semana_actual = cur.fetchone()[0] or 1

    def resolver_ejercicio(self, texto: str) -> EntidadEjercicio:
        texto_n = _norm(texto)
        candidatos = list(self._alias_a_canonico.keys())
        # 1) coincidencia por subcadena exacta (alias contenido literalmente en la pregunta)
        mejores = [a for a in candidatos if a in texto_n]
        if mejores:
            mejor = max(mejores, key=len)
            return EntidadEjercicio(self._alias_a_canonico[mejor], 100.0, mejor)
        # 2) fuzzy matching contra ventanas de texto (n-gramas de palabras) para tolerar
        #    erratas/variantes no listadas como alias. Se descartan ventanas triviales
        #    (stopwords sueltas o fragmentos muy cortos) para evitar falsos positivos:
        #    p.ej. "el" no debe "casi-coincidir" con "gemelos" por simple coincidencia
        #    de subcadena.
        palabras = [p for p in texto_n.split() if p not in STOPWORDS]
        mejor_resultado = None
        for n in (1, 2, 3, 4):
            for i in range(len(palabras) - n + 1):
                ventana = " ".join(palabras[i:i + n])
                if len(ventana) < 4:
                    continue
                resultado = process.extractOne(ventana, candidatos, scorer=fuzz.token_sort_ratio)
                if resultado and (mejor_resultado is None or resultado[1] > mejor_resultado[1]):
                    mejor_resultado = resultado
        if mejor_resultado is None:
            return EntidadEjercicio(None, 0.0)
        alias_encontrado, score, _ = mejor_resultado
        # Solo se da por resuelto con confianza alta (>= UMBRAL_ACEPTAR). Entre
        # UMBRAL_ACLARAR y UMBRAL_ACEPTAR se deja nombre_canonico=None pero se conserva
        # el score/candidato para que la capa de pipeline pueda pedir aclaracion en vez
        # de asumir un ejercicio con baja confianza.
        canonico = self._alias_a_canonico[alias_encontrado] if score >= UMBRAL_ACEPTAR else None
        return EntidadEjercicio(canonico, score, alias_encontrado)

    def resolver(self, texto: str) -> Entidades:
        texto_n = _norm(texto)
        ent = Entidades()

        ent.ejercicio = self.resolver_ejercicio(texto)

        for grupo, sinonimos in GRUPOS_SINONIMOS.items():
            if any(_norm(s) in texto_n for s in sinonimos):
                ent.grupo_muscular = grupo
                break

        for dia, sinonimos in DIAS_SINONIMOS.items():
            if any(_norm(s) in texto_n for s in sinonimos):
                ent.dia_semana = dia
                break

        for metrica, sinonimos in METRICA_SINONIMOS.items():
            if any(_norm(s) in texto_n for s in sinonimos):
                ent.metrica = metrica
                break

        for patron, valor in AGREGACION_PATRONES:
            if re.search(patron, texto_n):
                ent.agregacion = valor
                break

        for patron, valor in ORDINAL_PATRONES:
            if re.search(patron, texto_n):
                ent.ordinal = valor
                break

        # Periodos relativos y absolutos
        m = re.search(r"ultimas?\s+(\d+)\s+semanas?", texto_n)
        if m:
            n = int(m.group(1))
            ent.semana_fin = self.semana_actual
            ent.semana_inicio = max(1, self.semana_actual - n + 1)
        elif re.search(r"ultimo\s+mes|ultimas\s+cuatro\s+semanas", texto_n):
            ent.semana_fin = self.semana_actual
            ent.semana_inicio = max(1, self.semana_actual - 3)
        elif re.search(r"ultimos\s+tres\s+meses|todo\s+el\s+bloque|desde\s+el\s+principio", texto_n):
            ent.semana_fin = self.semana_actual
            ent.semana_inicio = 1
        elif re.search(r"primera\s+semana", texto_n):
            ent.semana = 1
        elif re.search(r"(ultima|última)\s+semana|esta\s+semana", texto_n):
            ent.semana = self.semana_actual
        else:
            m = re.search(r"semana\s+(\d+)", texto_n)
            if m:
                ent.semana = int(m.group(1))

        m = re.search(r"mas\s+de\s+(\d+([.,]\d+)?)\s*kg", texto_n)
        if m:
            ent.umbral = float(m.group(1).replace(",", "."))
            ent.umbral_operador = ">"

        return ent
