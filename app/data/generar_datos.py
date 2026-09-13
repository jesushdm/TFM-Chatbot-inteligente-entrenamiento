"""
Generacion de datos sinteticos para el asistente conversacional de entrenamiento.

Fuente ("basado en la plantilla", segun lo acordado con el usuario):
  - data/source/Rutina_entrenamiento.md   -> PLAN semanal (rutina_ejercicio)
  - data/source/Registro_historico.md     -> semanas 1-4 REALES, usadas tal cual
                                              y como semilla de estilo para generar
                                              las semanas 5-13 (hasta completar 3 meses)

Alcance acordado con el usuario (2026-09-13): 3 meses (13 semanas), no 18 meses,
porque la rutina real solo cubre 3 dias/semana (Lunes/Martes/Miercoles) y 4 semanas
de historico real. Se mantiene la logica de progresion + descarga periodica + ruido
gaussiano ya descrita en el Capitulo 3 de la Memoria, aplicada a esta escala menor.

Regla de generacion (documentada, simple a proposito -> "no complicarse
innecesariamente"):
  - Se definen mesociclos de 4 semanas. La posicion dentro del mesociclo (1..4)
    reutiliza el patron de repeticiones y RIR observado en las semanas 1-4 reales
    (semana con posicion 4 = semana de descarga: menos repeticiones, RIR mas alto).
  - El peso progresa cada semana (excepto en las semanas de descarga, donde se
    mantiene) usando el incremento medio observado entre las semanas 1-2 y 2-3 del
    historico real de cada ejercicio, con un pequeno ruido gaussiano (semilla fija,
    reproducible) y redondeo a multiplos de 0.5 kg.
  - El ejercicio "Plancha abdominal" es isometrico: no se genera con peso, y no se
    registra en el historico (igual que en Registro_historico.md, donde no aparece),
    aunque si figura como programado en el plan semanal.

Limitacion reconocida (ya recogida en el apartado de limitaciones del Cap.3): no se
modela ruido de sesiones perdidas, series fallidas ni sustituciones de ejercicio;
se ha omitido a proposito para mantener el generador simple en esta fase.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

SOURCE_DIR = Path(__file__).parent / "source"
RUTINA_MD = SOURCE_DIR / "Rutina_entrenamiento.md"
HISTORICO_MD = SOURCE_DIR / "Registro_historico.md"

SEED = 42
NUM_SEMANAS = 13  # ~3 meses (13 x 7 dias = 91 dias)
DIAS_ENTRENO = ["Lunes", "Martes", "Miercoles"]

# Fecha de referencia declarada (fija, reproducible): lunes de la semana 13 (la mas
# reciente). No se usa la fecha real del sistema para que el dataset generado sea
# siempre el mismo con la misma semilla.
FECHA_REFERENCIA = date(2026, 9, 14)  # lunes
FECHA_INICIO_BLOQUE = FECHA_REFERENCIA - timedelta(weeks=NUM_SEMANAS - 1)

# Mapa dia -> desplazamiento en dias respecto al lunes de cada semana
OFFSET_DIA = {"Lunes": 0, "Martes": 1, "Miercoles": 2}

# Clasificacion por grupo muscular. No estaba en los archivos del usuario; se asigna
# con un criterio anatomico estandar (ver nota de implementacion en la Memoria).
GRUPO_MUSCULAR = {
    "Press banca con barra": "Pecho",
    "Remo con barra": "Espalda",
    "Press militar": "Hombro",
    "Jalon al pecho": "Espalda",
    "Curl de biceps": "Biceps",
    "Extension de triceps en polea": "Triceps",
    "Sentadilla con barra": "Piernas",
    "Peso muerto rumano": "Piernas",
    "Prensa de piernas": "Piernas",
    "Curl femoral": "Piernas",
    "Elevacion de gemelos": "Piernas",
    "Plancha abdominal": "Core",
    "Press inclinado con mancuernas": "Pecho",
    "Dominadas / jalon al pecho": "Espalda",
    "Remo sentado en polea": "Espalda",
    "Elevaciones laterales": "Hombro",
    "Curl de biceps con mancuernas": "Biceps",
    "Extension de triceps": "Triceps",
}

EQUIPAMIENTO = {
    "Press banca con barra": "barra",
    "Remo con barra": "barra",
    "Press militar": "barra",
    "Jalon al pecho": "polea",
    "Curl de biceps": "barra",
    "Extension de triceps en polea": "polea",
    "Sentadilla con barra": "barra",
    "Peso muerto rumano": "barra",
    "Prensa de piernas": "maquina",
    "Curl femoral": "maquina",
    "Elevacion de gemelos": "maquina",
    "Plancha abdominal": "peso corporal",
    "Press inclinado con mancuernas": "mancuernas",
    "Dominadas / jalon al pecho": "polea",
    "Remo sentado en polea": "polea",
    "Elevaciones laterales": "mancuernas",
    "Curl de biceps con mancuernas": "mancuernas",
    "Extension de triceps": "polea",
}

# Alias cortos habituales para la resolucion de entidades (ademas del nombre completo,
# que siempre es un alias implicito). Solo se incluyen cuando no son ambiguos.
ALIAS_EXTRA = {
    "Press banca con barra": ["press banca", "banca"],
    "Remo con barra": ["remo con barra", "remo barra"],
    "Press militar": ["press militar", "militar"],
    "Jalon al pecho": ["jalon al pecho", "jalon pecho", "jalones"],
    "Curl de biceps": ["curl biceps", "curl de biceps con barra"],
    "Extension de triceps en polea": ["extension triceps polea", "triceps en polea"],
    "Sentadilla con barra": ["sentadilla", "sentadillas"],
    "Peso muerto rumano": ["peso muerto", "muerto rumano", "muerto"],
    "Prensa de piernas": ["prensa", "prensa piernas"],
    "Curl femoral": ["femoral", "curl femoral"],
    "Elevacion de gemelos": ["gemelos", "elevacion gemelos"],
    "Plancha abdominal": ["plancha", "plancha abdominal"],
    "Press inclinado con mancuernas": ["press inclinado", "inclinado con mancuernas"],
    "Dominadas / jalon al pecho": ["dominadas"],
    "Remo sentado en polea": ["remo sentado", "remo en polea"],
    "Elevaciones laterales": ["elevaciones laterales", "laterales"],
    "Curl de biceps con mancuernas": ["curl biceps mancuernas", "curl mancuernas"],
    "Extension de triceps": ["extension triceps"],
}

_ES_ISOMETRICO = {"Plancha abdominal"}


def _norm(txt: str) -> str:
    """Normaliza acentos/espacios de las tablas markdown fuente a ASCII simple."""
    txt = txt.strip()
    repl = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
    }
    for a, b in repl.items():
        txt = txt.replace(a, b)
    return txt


def _parse_markdown_tables(md_text: str):
    """Devuelve una lista de (encabezado_seccion_o_None, [fila_dict, ...]) por cada
    tabla markdown encontrada, asociando cada tabla al ultimo encabezado '## ...' visto."""
    lines = md_text.splitlines()
    tablas = []
    encabezado_actual = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("## "):
            encabezado_actual = _norm(line[3:].strip())
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            headers = [_norm(h) for h in line.strip("|").split("|")]
            headers = [h.strip() for h in headers]
            i += 2
            filas = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cols = [c.strip() for c in lines[i].strip("|").split("|")]
                if len(cols) == len(headers):
                    filas.append({h: _norm(c) for h, c in zip(headers, cols)})
                i += 1
            tablas.append((encabezado_actual, filas))
            continue
        i += 1
    return tablas


@dataclass
class FilaPlan:
    dia: str
    ejercicio: str
    series: int
    rep_min: int
    rep_max: int
    rir: int


@dataclass
class FilaHistorico:
    semana: int
    dia: str
    ejercicio: str
    series: int
    repeticiones: int
    rir: int
    peso_kg: float | None  # None para ejercicios isometricos


def cargar_plan() -> list[FilaPlan]:
    # Rutina_entrenamiento.md no tiene columna "Dia" en la tabla: el dia viene del
    # encabezado de seccion "## Lunes - Tren superior A" que precede a cada tabla.
    texto = RUTINA_MD.read_text(encoding="utf-8")
    filas = []
    for encabezado, tabla in _parse_markdown_tables(texto):
        if not encabezado:
            continue
        dia = encabezado.split()[0]
        for f in tabla:
            reps = f["Repeticiones"]
            m = re.match(r"(\d+)\D+(\d+)", reps)
            if m:
                rmin, rmax = int(m.group(1)), int(m.group(2))
            else:
                # caso "30-45 s" (isometrico): se guarda igualmente como rango numerico
                nums = re.findall(r"\d+", reps)
                rmin, rmax = int(nums[0]), int(nums[-1])
            filas.append(FilaPlan(
                dia=dia,
                ejercicio=f["Ejercicio"],
                series=int(f["Series"]),
                rep_min=rmin,
                rep_max=rmax,
                rir=int(f["RIR"]),
            ))
    return filas


def cargar_historico() -> list[FilaHistorico]:
    texto = HISTORICO_MD.read_text(encoding="utf-8")
    filas = []
    for _, tabla in _parse_markdown_tables(texto):
        for f in tabla:
            peso_txt = f["Peso"]
            peso = None
            if peso_txt:
                num = peso_txt.replace("kg", "").strip().replace(",", ".")
                peso = float(num)
            filas.append(FilaHistorico(
                semana=int(f["Semana"]),
                dia=f["Dia"],
                ejercicio=f["Ejercicio"],
                series=int(f["Series"]),
                repeticiones=int(f["Repeticiones"]),
                rir=int(f["RIR"]),
                peso_kg=peso,
            ))
    return filas


def fecha_de(semana: int, dia: str) -> date:
    lunes_semana = FECHA_INICIO_BLOQUE + timedelta(weeks=semana - 1)
    return lunes_semana + timedelta(days=OFFSET_DIA[dia])


@dataclass
class DatosGenerados:
    ejercicios: list[dict] = field(default_factory=list)          # {nombre, grupo_muscular, equipamiento, unilateral, es_isometrico}
    alias: list[tuple[str, str]] = field(default_factory=list)    # (ejercicio, alias)
    plan: list[FilaPlan] = field(default_factory=list)
    sesiones: list[dict] = field(default_factory=list)            # {semana, dia, fecha, duracion_min, rpe_sesion}
    series_por_sesion: dict = field(default_factory=dict)         # (semana,dia) -> [FilaHistorico-like]


def generar(seed: int = SEED, num_semanas: int = NUM_SEMANAS) -> DatosGenerados:
    rnd = random.Random(seed)
    plan = cargar_plan()
    historico_real = cargar_historico()  # semanas 1-4, tal cual

    nombres = sorted({p.ejercicio for p in plan} | {h.ejercicio for h in historico_real})
    datos = DatosGenerados()
    for nombre in nombres:
        datos.ejercicios.append({
            "nombre": nombre,
            "grupo_muscular": GRUPO_MUSCULAR.get(nombre, "Otro"),
            "equipamiento": EQUIPAMIENTO.get(nombre, "otro"),
            "unilateral": 0,
            "es_isometrico": 1 if nombre in _ES_ISOMETRICO else 0,
        })
        for a in ALIAS_EXTRA.get(nombre, []):
            datos.alias.append((nombre, a))
    datos.plan = plan

    # --- Incremento medio semanal por ejercicio, calculado sobre las semanas 1-3 reales ---
    por_ejercicio_dia = {}
    for h in historico_real:
        por_ejercicio_dia.setdefault((h.ejercicio, h.dia), []).append(h)
    incremento_medio = {}
    for (ejercicio, dia), filas in por_ejercicio_dia.items():
        filas = sorted(filas, key=lambda x: x.semana)
        pesos_1_3 = [f.peso_kg for f in filas if f.semana in (1, 2, 3) and f.peso_kg is not None]
        if len(pesos_1_3) >= 2:
            deltas = [pesos_1_3[i + 1] - pesos_1_3[i] for i in range(len(pesos_1_3) - 1)]
            incremento_medio[(ejercicio, dia)] = sum(deltas) / len(deltas)
        else:
            incremento_medio[(ejercicio, dia)] = 0.0

    # Patron reps/RIR por posicion en el mesociclo (1..4), tomado directamente de las
    # semanas reales 1-4.
    patron_pos = {1: {}, 2: {}, 3: {}, 4: {}}
    for h in historico_real:
        patron_pos[h.semana][(h.ejercicio, h.dia)] = (h.repeticiones, h.rir)

    ultimo_peso = {}  # (ejercicio, dia) -> peso de la semana anterior

    for semana in range(1, num_semanas + 1):
        posicion = ((semana - 1) % 4) + 1
        for dia in DIAS_ENTRENO:
            fecha = fecha_de(semana, dia)
            filas_dia = []
            for fp in [p for p in plan if p.dia == dia]:
                clave = (fp.ejercicio, dia)
                es_isometrico = fp.ejercicio in _ES_ISOMETRICO

                if semana <= 4:
                    # Semanas reales: se copian tal cual del historico (si existen).
                    real = next((h for h in historico_real
                                 if h.semana == semana and h.dia == dia and h.ejercicio == fp.ejercicio), None)
                    if real is None:
                        # No registrado en el historico real (caso: Plancha abdominal)
                        continue
                    reps, rir, peso = real.repeticiones, real.rir, real.peso_kg
                else:
                    if es_isometrico:
                        continue  # no se genera historico para ejercicios isometricos
                    reps, rir = patron_pos[posicion].get(clave, (fp.rep_max, fp.rir))
                    inc = incremento_medio.get(clave, 0.0)
                    base = ultimo_peso.get(clave)
                    if base is None:
                        # no deberia ocurrir (toda clave del plan tiene semanas 1-4 reales)
                        base = 20.0
                    if posicion == 4:
                        peso = base  # semana de descarga: peso se mantiene
                    else:
                        ruido = rnd.gauss(0, 0.3)
                        peso = base + inc + ruido
                    peso = round(peso * 2) / 2  # redondeo a 0.5 kg

                ultimo_peso[clave] = peso
                filas_dia.append(FilaHistorico(
                    semana=semana, dia=dia, ejercicio=fp.ejercicio,
                    series=fp.series, repeticiones=reps, rir=rir, peso_kg=peso,
                ))

            if not filas_dia:
                continue
            rir_medio = sum(f.rir for f in filas_dia) / len(filas_dia)
            duracion = 50 + 3 * len(filas_dia) + rnd.randint(-5, 8)
            rpe = round(min(10.0, max(5.0, 9.5 - rir_medio + rnd.gauss(0, 0.2))), 1)
            datos.sesiones.append({
                "semana": semana, "dia": dia, "fecha": fecha.isoformat(),
                "duracion_min": duracion, "rpe_sesion": rpe,
            })
            datos.series_por_sesion[(semana, dia)] = filas_dia

    return datos


if __name__ == "__main__":
    d = generar()
    print(f"Ejercicios: {len(d.ejercicios)}")
    print(f"Filas de plan (rutina_ejercicio): {len(d.plan)}")
    print(f"Sesiones generadas: {len(d.sesiones)}")
    total_series = sum(len(v) for v in d.series_por_sesion.values())
    print(f"Series generadas: {total_series}")
