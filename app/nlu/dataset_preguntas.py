"""
Construccion del dataset anotado de preguntas (apartado 3.5 de la Memoria).

Sigue, de forma simplificada y automatizada ("lo mas sencillo posible", segun lo
acordado con el usuario), el procedimiento de seis pasos ya documentado en el Cap.3:
  1) inventario de preguntas respondibles por interseccion x entidad,
  2-3) plantillas parametrizadas escritas a mano y rellenadas con valores del
       catalogo (ejercicios, grupos, dias, periodos),
  4) paráfrasis: en vez de generarlas con un LLM y revisarlas una a una (como
     describe el Cap.3 para la version de investigacion completa), aqui se anaden
     variantes de fraseo escritas a mano por plantilla -> ver nota de alcance en la
     Memoria,
  5) ruido linguistico controlado: se inyecta automaticamente (mayusculas/minusculas,
     tildes, una errata simple) sobre una fraccion de los ejemplos, con semilla fija,
  6) anotacion de los 5 campos: intencion, entidades BIO, forma logica, SQL de
     referencia y respuesta esperada -- estos tres ultimos se calculan ejecutando de
     verdad la forma logica contra tfm.db, reutilizando core/sql_templates.py y
     core/responder.py, para que el dataset y el motor de respuesta nunca diverjan.

La particion train/val/test (70/15/15) se hace por plantilla de origen (no al azar
por ejemplo), y se reserva ademas un conjunto fuera de distribucion (OOD).
"""
from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.db import conectar_solo_lectura  # noqa: E402
from core.forma_logica import Ambito, FormaLogica  # noqa: E402
from core.responder import responder  # noqa: E402
from core.sql_templates import ejecutar  # noqa: E402

SEED = 42
OUT_DIR = ROOT / "nlu" / "dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚáéíóúÑñ0-9]+")


def _cyclo(lista, i):
    """Elige un elemento de forma deterministica pero variada (evita el producto
    cartesiano completo entre plantillas x catalogo x periodos, que dispararia el
    tamano del dataset sin anadir variedad real)."""
    return lista[i % len(lista)]


# --------------------------------------------------------------------------------
# Utilidades de plantilla -> texto + spans de entidad -> BIO
# --------------------------------------------------------------------------------

def rellenar(template: str, valores: dict) -> tuple[str, list[tuple[int, int, str]]]:
    """valores: campo -> (texto_reemplazo, tipo_entidad_o_None)."""
    spans = []
    piezas = []
    ultimo = 0
    for m in re.finditer(r"\{(\w+)\}", template):
        piezas.append(template[ultimo:m.start()])
        campo = m.group(1)
        texto_val, tipo = valores[campo]
        inicio = sum(len(x) for x in piezas)
        piezas.append(texto_val)
        fin = inicio + len(texto_val)
        if tipo:
            spans.append((inicio, fin, tipo))
        ultimo = m.end()
    piezas.append(template[ultimo:])
    return "".join(piezas), spans


def bio_tags(texto: str, spans: list[tuple[int, int, str]]) -> list[tuple[str, str]]:
    etiquetas = []
    for m in TOKEN_RE.finditer(texto):
        ts, te, tok = m.start(), m.end(), m.group(0)
        tag = "O"
        for (s, e, tipo) in spans:
            if ts >= s and te <= e:
                tag = ("B-" if ts == s else "I-") + tipo
                break
        etiquetas.append((tok, tag))
    return etiquetas


# --------------------------------------------------------------------------------
# Ruido linguistico controlado (paso 5), aplicado a una fraccion de los ejemplos
# --------------------------------------------------------------------------------

def _quitar_tildes(txt: str) -> str:
    tabla = str.maketrans("áéíóúÁÉÍÓÚ", "aeiouAEIOU")
    return txt.translate(tabla)


def _errata_simple(txt: str, rnd: random.Random) -> str:
    palabras = txt.split(" ")
    largas = [i for i, p in enumerate(palabras) if len(p) > 5]
    if not largas:
        return txt
    i = rnd.choice(largas)
    p = list(palabras[i])
    j = rnd.randrange(len(p) - 1)
    p[j], p[j + 1] = p[j + 1], p[j]
    palabras[i] = "".join(p)
    return " ".join(palabras)


def aplicar_ruido(texto: str, rnd: random.Random) -> str:
    r = rnd.random()
    if r < 0.12:
        texto = texto.replace("¿", "").replace("?", "")
    if r < 0.10:
        texto = _quitar_tildes(texto)
    if 0.20 <= r < 0.28:
        texto = _errata_simple(texto, rnd)
    if r < 0.05:
        texto = texto.lower()
    return texto


# --------------------------------------------------------------------------------
# Catalogo (consultado en vivo a la BD, para no duplicar listas hardcodeadas)
# --------------------------------------------------------------------------------

@dataclass
class Catalogo:
    ejercicios_con_peso: list[str]
    ejercicios_todos: list[str]
    grupos: list[str]
    dias: list[str]
    semana_actual: int


def cargar_catalogo(con) -> Catalogo:
    cur = con.execute("SELECT nombre FROM ejercicio WHERE es_isometrico = 0 ORDER BY nombre")
    con_peso = [r[0] for r in cur.fetchall()]
    cur = con.execute("SELECT nombre FROM ejercicio ORDER BY nombre")
    todos = [r[0] for r in cur.fetchall()]
    cur = con.execute("SELECT DISTINCT grupo_muscular FROM ejercicio ORDER BY grupo_muscular")
    grupos = [r[0] for r in cur.fetchall()]
    cur = con.execute("SELECT MAX(semana) FROM sesion")
    semana_actual = cur.fetchone()[0]
    return Catalogo(con_peso, todos, grupos, ["Lunes", "Martes", "Miercoles"], semana_actual)


def periodos_rango(cat: Catalogo):
    s = cat.semana_actual
    return [
        ("las últimas 2 semanas", max(1, s - 1), s),
        ("las últimas 4 semanas", max(1, s - 3), s),
        ("las últimas 6 semanas", max(1, s - 5), s),
        ("el último mes", max(1, s - 3), s),
        ("los últimos tres meses", 1, s),
        ("todo el bloque de entrenamiento", 1, s),
    ]


def periodos_puntuales(cat: Catalogo):
    s = cat.semana_actual
    return [
        ("la semana 1", 1),
        ("la primera semana", 1),
        (f"la semana {max(1, s - 6)}", max(1, s - 6)),
        ("la última semana", s),
        (f"la semana {s}", s),
    ]


# --------------------------------------------------------------------------------
# Definicion de ejemplos por intencion
# --------------------------------------------------------------------------------

@dataclass
class Ejemplo:
    texto: str
    intencion: str
    plantilla_id: str
    forma_logica: FormaLogica
    bio: list[tuple[str, str]] = field(default_factory=list)
    sql_referencia: str = ""
    respuesta_esperada: str = ""
    con_ruido: bool = False


def _fl_a_dict(fl: FormaLogica) -> dict:
    d = asdict(fl)
    return d


def _generar_consulta_progresion(cat: Catalogo, rnd: random.Random) -> list[Ejemplo]:
    plantillas = [
        "¿Cuánto he progresado en {ejercicio} durante {periodo}?",
        "¿Cómo ha evolucionado mi {ejercicio} durante {periodo}?",
        "¿Cómo ha cambiado el peso de {ejercicio} en {periodo}?",
        "Evolución de {ejercicio} en {periodo}",
        "¿He mejorado en {ejercicio} en {periodo}?",
        "Dame la progresión de {ejercicio} en {periodo}",
        "¿Cuál ha sido mi evolución de fuerza en {ejercicio} durante {periodo}?",
    ]
    periodos = periodos_rango(cat)
    ejemplos = []
    for pid, plantilla in enumerate(plantillas):
        for idx, ejercicio in enumerate(cat.ejercicios_con_peso):
            periodo_txt, ini, fin = _cyclo(periodos, pid + idx)
            texto, spans = rellenar(plantilla, {
                "ejercicio": (ejercicio, "EJERCICIO"),
                "periodo": (periodo_txt, "PERIODO"),
            })
            fl = FormaLogica("consulta_progresion", metrica="peso",
                              ambito=Ambito(ejercicio=ejercicio, semana_inicio=ini, semana_fin=fin))
            ejemplos.append(Ejemplo(texto, "consulta_progresion", f"progresion_{pid}", fl, bio_tags(texto, spans)))

    # Rankings sin ejercicio concreto (top exercises por delta)
    for texto, periodo_txt, ini, fin in [
        ("¿Qué ejercicios han aumentado más de peso?", "las últimas 4 semanas", max(1, cat.semana_actual - 3), cat.semana_actual),
        ("¿Qué ejercicios muestran una progresión de fuerza más clara?", "el bloque completo", 1, cat.semana_actual),
        ("¿Qué ejercicios han mantenido el mismo peso durante las últimas 4 semanas?", "las últimas 4 semanas", max(1, cat.semana_actual - 3), cat.semana_actual),
    ]:
        fl = FormaLogica("consulta_progresion", metrica="peso", top_n=5,
                          ambito=Ambito(semana_inicio=ini, semana_fin=fin))
        ejemplos.append(Ejemplo(texto, "consulta_progresion", "progresion_ranking", fl, bio_tags(texto, [])))
    return ejemplos


def _generar_consulta_maximo(cat: Catalogo, rnd: random.Random) -> list[Ejemplo]:
    plantillas_peso = [
        "¿Cuál ha sido mi mayor peso registrado en {ejercicio}?",
        "¿Cuál es mi mejor marca en {ejercicio}?",
        "¿Cuánto es lo máximo que he levantado en {ejercicio}?",
        "¿Cuál es mi récord de peso en {ejercicio}?",
        "¿Cuál ha sido el peso más alto que he movido en {ejercicio}?",
    ]
    ejemplos = []
    for pid, plantilla in enumerate(plantillas_peso):
        for ejercicio in cat.ejercicios_con_peso:
            texto, spans = rellenar(plantilla, {"ejercicio": (ejercicio, "EJERCICIO")})
            fl = FormaLogica("consulta_maximo", metrica="peso", agregacion="maximo", ambito=Ambito(ejercicio=ejercicio))
            ejemplos.append(Ejemplo(texto, "consulta_maximo", f"maximo_peso_{pid}", fl, bio_tags(texto, spans)))

    for texto, metrica, agregacion in [
        ("¿Qué semana tuvo mayor volumen de entrenamiento?", "volumen", "maximo"),
        ("¿Cuál fue la semana con menor intensidad?", "rpe", "minimo"),
        ("¿Cuál fue mi semana de entrenamiento más exigente?", "rpe", "maximo"),
        ("¿Cuál ha sido mi semana con más tonelaje levantado?", "tonelaje", "maximo"),
    ]:
        fl = FormaLogica("consulta_maximo", metrica=metrica, agregacion=agregacion, ambito=Ambito())
        ejemplos.append(Ejemplo(texto, "consulta_maximo", "maximo_semana", fl, bio_tags(texto, [])))
    return ejemplos


def _generar_consulta_volumen(cat: Catalogo, rnd: random.Random) -> list[Ejemplo]:
    plantillas = [
        "¿Cuántas series de {ejercicio} hice en {periodo}?",
        "¿Cuánto volumen de {ejercicio} he hecho en {periodo}?",
        "¿Cuántas series efectivas de {ejercicio} acumulé en {periodo}?",
        "¿Qué volumen total he movido en {ejercicio} durante {periodo}?",
    ]
    periodos = periodos_rango(cat)
    ejemplos = []
    for pid, plantilla in enumerate(plantillas):
        for idx, ejercicio in enumerate(cat.ejercicios_con_peso):
            periodo_txt, ini, fin = _cyclo(periodos, pid + idx)
            texto, spans = rellenar(plantilla, {
                "ejercicio": (ejercicio, "EJERCICIO"), "periodo": (periodo_txt, "PERIODO"),
            })
            fl = FormaLogica("consulta_volumen", ambito=Ambito(ejercicio=ejercicio, semana_inicio=ini, semana_fin=fin))
            ejemplos.append(Ejemplo(texto, "consulta_volumen", f"volumen_ejercicio_{pid}", fl, bio_tags(texto, spans)))

    plantillas_grupo = [
        "¿Cuántas series de {grupo} hice en {periodo}?",
        "¿Cuánto volumen de {grupo} acumulé en {periodo}?",
    ]
    for pid, plantilla in enumerate(plantillas_grupo):
        for idx, grupo in enumerate(cat.grupos):
            periodo_txt, ini, fin = _cyclo(periodos, pid + idx)
            texto, spans = rellenar(plantilla, {
                "grupo": (grupo.lower(), "GRUPO"), "periodo": (periodo_txt, "PERIODO"),
            })
            fl = FormaLogica("consulta_volumen", ambito=Ambito(grupo_muscular=grupo, semana_inicio=ini, semana_fin=fin))
            ejemplos.append(Ejemplo(texto, "consulta_volumen", f"volumen_grupo_{pid}", fl, bio_tags(texto, spans)))

    for texto in ["¿Cuál ha sido mi evolución del volumen de entrenamiento?",
                  "¿Cómo ha evolucionado mi volumen de entrenamiento semana a semana?",
                  "¿Cómo ha ido variando mi volumen total semana a semana?"]:
        fl = FormaLogica("consulta_volumen", ambito=Ambito(semana_inicio=1, semana_fin=cat.semana_actual))
        ejemplos.append(Ejemplo(texto, "consulta_volumen", "volumen_evolucion", fl, bio_tags(texto, [])))
    return ejemplos


def _generar_consulta_frecuencia(cat: Catalogo, rnd: random.Random) -> list[Ejemplo]:
    ejemplos = []
    plantillas = [
        "¿Cuántas veces entrené {grupo} en {periodo}?",
        "¿Cuántas sesiones de {grupo} he hecho en {periodo}?",
        "¿Cuántas veces he trabajado {grupo} en {periodo}?",
    ]
    periodos = periodos_rango(cat)
    for pid, plantilla in enumerate(plantillas):
        for idx, grupo in enumerate(cat.grupos):
            periodo_txt, ini, fin = _cyclo(periodos, pid + idx)
            texto, spans = rellenar(plantilla, {
                "grupo": (grupo.lower(), "GRUPO"), "periodo": (periodo_txt, "PERIODO"),
            })
            fl = FormaLogica("consulta_frecuencia", ambito=Ambito(grupo_muscular=grupo, semana_inicio=ini, semana_fin=fin))
            ejemplos.append(Ejemplo(texto, "consulta_frecuencia", f"frecuencia_grupo_{pid}", fl, bio_tags(texto, spans)))

    for plantilla in ["¿Cuántas sesiones he entrenado en {periodo}?", "¿Cuántos entrenamientos he hecho en {periodo}?"]:
        for periodo_txt, ini, fin in periodos:
            texto, spans = rellenar(plantilla, {"periodo": (periodo_txt, "PERIODO")})
            fl = FormaLogica("consulta_frecuencia", ambito=Ambito(semana_inicio=ini, semana_fin=fin))
            ejemplos.append(Ejemplo(texto, "consulta_frecuencia", "frecuencia_total", fl, bio_tags(texto, spans)))

    plantillas_ejercicio = [
        "¿Cuántas veces he hecho {ejercicio} en {periodo}?",
        "¿En cuántas sesiones he entrenado {ejercicio} durante {periodo}?",
    ]
    for pid, plantilla in enumerate(plantillas_ejercicio):
        for idx, ejercicio in enumerate(cat.ejercicios_con_peso):
            periodo_txt, ini, fin = _cyclo(periodos, pid + idx)
            texto, spans = rellenar(plantilla, {
                "ejercicio": (ejercicio, "EJERCICIO"), "periodo": (periodo_txt, "PERIODO"),
            })
            fl = FormaLogica("consulta_frecuencia", ambito=Ambito(ejercicio=ejercicio, semana_inicio=ini, semana_fin=fin))
            ejemplos.append(Ejemplo(texto, "consulta_frecuencia", f"frecuencia_ejercicio_{pid}", fl, bio_tags(texto, spans)))
    return ejemplos


def _generar_consulta_registro(cat: Catalogo, rnd: random.Random) -> list[Ejemplo]:
    plantillas = [
        "¿Qué peso utilicé en {ejercicio} en {periodo}?",
        "¿Cuánto levanté en {ejercicio} en {periodo}?",
        "¿Con qué peso hice {ejercicio} en {periodo}?",
        "¿Cuántas repeticiones hice de {ejercicio} en {periodo}?",
    ]
    periodos = periodos_puntuales(cat)
    ejemplos = []
    for pid, plantilla in enumerate(plantillas):
        for idx, ejercicio in enumerate(cat.ejercicios_con_peso):
            periodo_txt, semana = _cyclo(periodos, pid + idx)
            texto, spans = rellenar(plantilla, {
                "ejercicio": (ejercicio, "EJERCICIO"), "periodo": (periodo_txt, "PERIODO"),
            })
            fl = FormaLogica("consulta_registro", ambito=Ambito(ejercicio=ejercicio, semana=semana))
            ejemplos.append(Ejemplo(texto, "consulta_registro", f"registro_{pid}", fl, bio_tags(texto, spans)))

    for ejercicio in cat.ejercicios_con_peso:
        texto, spans = rellenar("¿Qué peso utilicé en {ejercicio} en cada una de las últimas 4 semanas?",
                                 {"ejercicio": (ejercicio, "EJERCICIO")})
        fl = FormaLogica("consulta_registro",
                          ambito=Ambito(ejercicio=ejercicio, semana_inicio=max(1, cat.semana_actual - 3), semana_fin=cat.semana_actual))
        ejemplos.append(Ejemplo(texto, "consulta_registro", "registro_serie", fl, bio_tags(texto, spans)))
    return ejemplos


def _generar_comparacion(cat: Catalogo, rnd: random.Random) -> list[Ejemplo]:
    ejemplos = []
    s = cat.semana_actual
    plantillas = [
        "¿Cuánto ha aumentado el peso de {ejercicio} desde la primera semana?",
        "¿Cómo ha cambiado {ejercicio} entre la primera y la última semana?",
    ]
    for pid, plantilla in enumerate(plantillas):
        for ejercicio in cat.ejercicios_con_peso:
            texto, spans = rellenar(plantilla, {"ejercicio": (ejercicio, "EJERCICIO")})
            fl = FormaLogica("comparacion", ambito=Ambito(ejercicio=ejercicio, semana=1),
                              comparar_con=Ambito(ejercicio=ejercicio, semana=s))
            ejemplos.append(Ejemplo(texto, "comparacion", f"comparacion_ejercicio_{pid}", fl, bio_tags(texto, spans)))

    for grupo in cat.grupos:
        texto, spans = rellenar("¿He hecho más volumen de {grupo} esta semana que la anterior?",
                                 {"grupo": (grupo.lower(), "GRUPO")})
        fl = FormaLogica("comparacion", ambito=Ambito(grupo_muscular=grupo, semana=s - 1),
                          comparar_con=Ambito(grupo_muscular=grupo, semana=s))
        ejemplos.append(Ejemplo(texto, "comparacion", "comparacion_grupo", fl, bio_tags(texto, spans)))

    for texto, sem_a, sem_b in [
        ("¿Qué diferencias hay entre mi entrenamiento de la primera y la cuarta semana?", 1, 4),
        ("¿Qué diferencias hay entre mi primera semana y mi última semana de entrenamiento?", 1, s),
        ("¿Cómo se compara mi semana 4 con mi semana 8?", 4, 8),
    ]:
        fl = FormaLogica("comparacion", ambito=Ambito(semana=sem_a), comparar_con=Ambito(semana=sem_b))
        ejemplos.append(Ejemplo(texto, "comparacion", "comparacion_semanas", fl, bio_tags(texto, [])))

    texto = "¿Levanté más volumen esta semana que la anterior?"
    fl = FormaLogica("comparacion", ambito=Ambito(semana=s - 1), comparar_con=Ambito(semana=s))
    ejemplos.append(Ejemplo(texto, "comparacion", "comparacion_semanas", fl, bio_tags(texto, [])))
    return ejemplos


def _generar_ultima_sesion(cat: Catalogo, rnd: random.Random) -> list[Ejemplo]:
    ejemplos = []
    for texto, ordinal in [
        ("¿Qué hice en mi último entrenamiento?", "ultima"),
        ("¿Cuál fue mi última sesión?", "ultima"),
        ("Resume mi última sesión de entrenamiento", "ultima"),
        ("¿Qué entrené la última vez?", "ultima"),
        ("Cuéntame cómo fue mi última sesión", "ultima"),
        ("¿Qué hice en mi penúltimo entrenamiento?", "penultima"),
        ("¿Qué entrené hace dos sesiones?", 2),
        ("¿Qué hice en mi tercer entrenamiento más reciente?", 3),
        ("¿Qué ejercicios hice en mi última sesión?", "ultima"),
        ("Dime el detalle de mi último entrenamiento", "ultima"),
        ("¿Cuánto duró mi última sesión?", "ultima"),
        ("¿Qué tal fue mi entrenamiento más reciente?", "ultima"),
        ("¿Qué hice la última vez que entrené?", "ultima"),
        ("Muéstrame mi sesión de entrenamiento más reciente", "ultima"),
        ("¿Qué series hice en mi penúltima sesión?", "penultima"),
        ("¿Cómo fue mi penúltimo entrenamiento?", "penultima"),
    ]:
        fl = FormaLogica("ultima_sesion", ordinal=ordinal, ambito=Ambito())
        ejemplos.append(Ejemplo(texto, "ultima_sesion", "ultima_sesion", fl, bio_tags(texto, [])))
    return ejemplos


def _generar_consulta_catalogo(cat: Catalogo, rnd: random.Random) -> list[Ejemplo]:
    plantillas = [
        "¿Qué ejercicios de {grupo} tengo registrados?",
        "¿Qué ejercicios de {grupo} he realizado?",
        "¿Qué ejercicios trabajan {grupo}?",
    ]
    ejemplos = []
    for pid, plantilla in enumerate(plantillas):
        for grupo in cat.grupos:
            texto, spans = rellenar(plantilla, {"grupo": (grupo.lower(), "GRUPO")})
            fl = FormaLogica("consulta_catalogo", ambito=Ambito(grupo_muscular=grupo))
            ejemplos.append(Ejemplo(texto, "consulta_catalogo", f"catalogo_grupo_{pid}", fl, bio_tags(texto, spans)))
    for texto in ["¿Qué ejercicios tengo registrados en el sistema?", "¿Qué ejercicios conoce el sistema?"]:
        fl = FormaLogica("consulta_catalogo", ambito=Ambito())
        ejemplos.append(Ejemplo(texto, "consulta_catalogo", "catalogo_todos", fl, bio_tags(texto, [])))
    return ejemplos


def _generar_consulta_programacion(cat: Catalogo, rnd: random.Random) -> list[Ejemplo]:
    ejemplos = []
    plantillas_dia = [
        "¿Qué ejercicios tengo programados para el {dia}?",
        "¿Qué toca entrenar el {dia}?",
        "¿Qué ejercicios me tocan el {dia}?",
    ]
    for pid, plantilla in enumerate(plantillas_dia):
        for dia in cat.dias:
            texto, spans = rellenar(plantilla, {"dia": (dia.lower(), "PERIODO")})
            fl = FormaLogica("consulta_programacion", ambito=Ambito(dia_semana=dia))
            ejemplos.append(Ejemplo(texto, "consulta_programacion", f"programacion_dia_{pid}", fl, bio_tags(texto, spans)))

    for dia in cat.dias:
        texto2, spans2 = rellenar("¿Cuántas series hago los {dia}?", {"dia": (_plural_dia(dia), "PERIODO")})
        fl2 = FormaLogica("consulta_programacion", metrica="series", ambito=Ambito(dia_semana=dia))
        ejemplos.append(Ejemplo(texto2, "consulta_programacion", "programacion_series_dia", fl2, bio_tags(texto2, spans2)))

    plantillas_ejercicio = [
        "¿Cuántas repeticiones tengo programadas para {ejercicio}?",
        "¿Cuántas series me tocan de {ejercicio} según el plan?",
        "¿En qué día tengo programado {ejercicio}?",
    ]
    for pid, plantilla in enumerate(plantillas_ejercicio):
        for ejercicio in cat.ejercicios_todos:
            texto, spans = rellenar(plantilla, {"ejercicio": (ejercicio, "EJERCICIO")})
            fl = FormaLogica("consulta_programacion", metrica="repeticiones", ambito=Ambito(ejercicio=ejercicio))
            ejemplos.append(Ejemplo(texto, "consulta_programacion", f"programacion_reps_{pid}", fl, bio_tags(texto, spans)))

    for texto, metrica in [
        ("¿Cuánto volumen semanal tengo programado?", "series"),
        ("¿Cuántas series tengo programadas en total a la semana?", "series"),
        ("¿Qué grupos musculares entreno cada día?", "grupo_muscular"),
        ("¿Qué grupos musculares trabajo cada día de la semana?", "grupo_muscular"),
        ("¿Qué grupo muscular toca cada día según mi plan?", "grupo_muscular"),
        ("¿Cuántos ejercicios diferentes hago durante la semana?", "ejercicios_distintos"),
        ("¿Cuántos ejercicios distintos tengo programados en la semana?", "ejercicios_distintos"),
        ("¿Cuántos ejercicios distintos incluye mi rutina?", "ejercicios_distintos"),
    ]:
        fl = FormaLogica("consulta_programacion", metrica=metrica, ambito=Ambito())
        ejemplos.append(Ejemplo(texto, "consulta_programacion", f"programacion_{metrica}", fl, bio_tags(texto, [])))
    return ejemplos


def _plural_dia(dia: str) -> str:
    return {"Lunes": "lunes", "Martes": "martes", "Miercoles": "miércoles"}[dia]


FUERA_DE_DOMINIO = [
    "¿Qué debería comer después de entrenar?",
    "¿Cuántas calorías tiene una pechuga de pollo?",
    "¿Qué suplementos me recomiendas para ganar masa muscular?",
    "¿Cuál es la capital de Francia?",
    "¿Qué tiempo va a hacer mañana?",
    "Cuéntame un chiste",
    "¿Cómo estiro antes de entrenar?",
    "¿Es mejor entrenar por la mañana o por la tarde?",
    "¿Qué proteína en polvo es mejor?",
    "¿Cuánta agua debo beber al día?",
    "¿Puedes recomendarme una dieta para definir?",
    "¿Qué opinas del ayuno intermitente?",
    "¿Qué película me recomiendas ver hoy?",
    "¿Cuánto cuesta un gimnasio en mi ciudad?",
    "¿Qué zapatillas son mejores para correr?",
    "¿Cómo se calienta antes de una carrera?",
    "¿Qué canción me recomiendas para entrenar?",
    "¿Cuántas horas hay que dormir para recuperar bien?",
    "Explícame qué es la creatina",
    "¿Qué app de entrenamiento me recomiendas?",
]

AMBIGUAS = [
    "¿Cuánto levanté en press?",
    "¿Cómo va mi progreso?",
    "¿Cuántas series hice?",
    "¿Cuál ha sido mi mejor marca?",
    "¿Qué tal la última semana?",
    "¿He mejorado?",
    "¿Cuánto peso hice en el ejercicio?",
    "¿Cómo ha ido el entrenamiento?",
    "¿Cuánto levanté en curl?",
    "¿Cuánto pesaba antes?",
    "¿Cuánto ha subido?",
    "¿Qué tal voy?",
    "¿Cuántas hice la semana pasada?",
    "¿Cuál fue el resultado?",
]


def _generar_control(cat: Catalogo) -> list[Ejemplo]:
    ejemplos = []
    for texto in FUERA_DE_DOMINIO:
        fl = FormaLogica("fuera_de_dominio")
        ejemplos.append(Ejemplo(texto, "fuera_de_dominio", "control_fdd", fl, bio_tags(texto, [])))
    for texto in AMBIGUAS:
        fl = FormaLogica("ambigua")
        ejemplos.append(Ejemplo(texto, "ambigua", "control_amb", fl, bio_tags(texto, [])))
    return ejemplos


GENERADORES = [
    _generar_consulta_progresion,
    _generar_consulta_maximo,
    _generar_consulta_volumen,
    _generar_consulta_frecuencia,
    _generar_consulta_registro,
    _generar_comparacion,
    _generar_ultima_sesion,
    _generar_consulta_catalogo,
    _generar_consulta_programacion,
]


def construir_dataset() -> list[Ejemplo]:
    con = conectar_solo_lectura()
    cat = cargar_catalogo(con)
    rnd = random.Random(SEED)

    ejemplos: list[Ejemplo] = []
    for gen in GENERADORES:
        ejemplos.extend(gen(cat, rnd))
    ejemplos.extend(_generar_control(cat))

    # Ejecutar cada forma logica contra la BD para anotar SQL de referencia y
    # respuesta esperada (fuente unica de verdad, comparte codigo con el motor real).
    pendientes = []
    for ej in ejemplos:
        if ej.intencion in ("fuera_de_dominio", "ambigua"):
            ej.sql_referencia = ""
            ej.respuesta_esperada = responder(ej.forma_logica, ejecutar(ej.forma_logica, con))
            pendientes.append(ej)
            continue
        try:
            res = ejecutar(ej.forma_logica, con)
        except Exception as exc:  # pragma: no cover - solo para depuracion
            print(f"[AVISO] fallo ejecutando forma logica de: {ej.texto!r} -> {exc}")
            continue
        if not res.filas:
            continue  # se descartan combinaciones sin datos (no aportan al dataset)
        ej.sql_referencia = res.sql
        ej.respuesta_esperada = responder(ej.forma_logica, res)
        pendientes.append(ej)
    ejemplos = pendientes

    # Ruido linguistico controlado sobre una fraccion (con semilla fija)
    for ej in ejemplos:
        original = ej.texto
        con_ruido = aplicar_ruido(ej.texto, rnd)
        if con_ruido != original:
            ej.texto = con_ruido
            ej.con_ruido = True
            ej.bio = bio_tags_reetiquetado(original, ej.bio, con_ruido)

    con.close()
    return ejemplos


def bio_tags_reetiquetado(original: str, bio_original: list[tuple[str, str]], nuevo_texto: str) -> list[tuple[str, str]]:
    """El ruido linguistico (quitar tildes/signos, una errata) no cambia el numero de
    tokens en la inmensa mayoria de los casos (solo transforma caracteres dentro de
    palabras existentes); se reetiqueta por posicion de token, y si el numero de
    tokens no coincide se recalculan solo las etiquetas O como fallback seguro."""
    tokens_nuevos = TOKEN_RE.findall(nuevo_texto)
    if len(tokens_nuevos) == len(bio_original):
        return [(tokens_nuevos[i], bio_original[i][1]) for i in range(len(tokens_nuevos))]
    return [(t, "O") for t in tokens_nuevos]


# --------------------------------------------------------------------------------
# Particion 70/15/15 por plantilla de origen + conjunto OOD
# --------------------------------------------------------------------------------

def particionar(ejemplos: list[Ejemplo], rnd: random.Random):
    """Particion 70/15/15 por plantilla de origen, ESTRATIFICADA POR INTENCION: se
    reparten las plantillas de cada intencion por separado y luego se combinan.

    Hacerlo sobre el conjunto global de plantillas (sin estratificar) puede dejar una
    interseccion entera fuera de un conjunto por puro azar, sobre todo las que tienen
    pocas plantillas distintas (p.ej. control_fdd/control_amb comparten una unica
    plantilla_id): se detecto exactamente ese problema durante la validacion (una
    interseccion con 0 ejemplos de test hundia el accuracy medido). Para intenciones
    con muy pocas plantillas (<=2) se reparte a nivel de EJEMPLO en vez de plantilla,
    para garantizar representacion en los tres conjuntos aun a costa de no separar
    perfectamente por plantilla en ese caso concreto.
    """
    por_interseccion: dict[str, dict[str, list[Ejemplo]]] = {}
    for ej in ejemplos:
        por_interseccion.setdefault(ej.intencion, {}).setdefault(ej.plantilla_id, []).append(ej)

    train, val, test = [], [], []
    for intencion, por_plantilla in por_interseccion.items():
        plantillas = sorted(por_plantilla.keys())
        rnd.shuffle(plantillas)
        n = len(plantillas)
        if n <= 2:
            ejemplos_intencion = [e for lst in por_plantilla.values() for e in lst]
            rnd.shuffle(ejemplos_intencion)
            n_e = len(ejemplos_intencion)
            n_test_e = max(1, round(n_e * 0.15))
            n_val_e = max(1, round(n_e * 0.15))
            test.extend(ejemplos_intencion[:n_test_e])
            val.extend(ejemplos_intencion[n_test_e:n_test_e + n_val_e])
            train.extend(ejemplos_intencion[n_test_e + n_val_e:])
            continue
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))
        test_ids = set(plantillas[:n_test])
        val_ids = set(plantillas[n_test:n_test + n_val])
        for pid, filas in por_plantilla.items():
            destino = test if pid in test_ids else (val if pid in val_ids else train)
            destino.extend(filas)
    return train, val, test


OOD_EJEMPLOS_TEXTO = [
    # Redactadas al final, sin usar las plantillas anteriores, con combinaciones no
    # vistas en el resto del dataset (p.ej. periodos en semanas no usadas antes).
    ("¿Cuánto ha subido mi press militar entre la semana 2 y la semana 11?", "comparacion",
     lambda cat: FormaLogica("comparacion", ambito=Ambito(ejercicio="Press militar", semana=2),
                              comparar_con=Ambito(ejercicio="Press militar", semana=11))),
    ("¿Cuál ha sido mi progresión en prensa de piernas a lo largo de todo el bloque?", "consulta_progresion",
     lambda cat: FormaLogica("consulta_progresion", metrica="peso",
                              ambito=Ambito(ejercicio="Prensa de piernas", semana_inicio=1, semana_fin=cat.semana_actual))),
    ("¿Qué ejercicios de espalda he realizado y cómo ha evolucionado el peso utilizado?", "consulta_catalogo",
     lambda cat: FormaLogica("consulta_catalogo", ambito=Ambito(grupo_muscular="Espalda"))),
    ("¿Cuántas series efectivas de pierna acumulé en las últimas seis semanas?", "consulta_volumen",
     lambda cat: FormaLogica("consulta_volumen", ambito=Ambito(grupo_muscular="Piernas",
                                                                 semana_inicio=max(1, cat.semana_actual - 5),
                                                                 semana_fin=cat.semana_actual))),
    ("¿Qué toca entrenar los miércoles?", "consulta_programacion",
     lambda cat: FormaLogica("consulta_programacion", ambito=Ambito(dia_semana="Miercoles"))),
    ("¿Qué remedio casero me quita las agujetas?", "fuera_de_dominio", lambda cat: FormaLogica("fuera_de_dominio")),
    ("¿Ha mejorado mi marca?", "ambigua", lambda cat: FormaLogica("ambigua")),
]


def construir_ood(cat: Catalogo, con) -> list[Ejemplo]:
    ejemplos = []
    for texto, intencion, fabrica in OOD_EJEMPLOS_TEXTO:
        fl = fabrica(cat)
        if fl.intencion in ("fuera_de_dominio", "ambigua"):
            res = ejecutar(fl, con)
        else:
            res = ejecutar(fl, con)
        respuesta = responder(fl, res)
        ejemplos.append(Ejemplo(texto, intencion, "ood", fl, bio_tags(texto, []), res.sql if res.sql else "", respuesta))
    return ejemplos


def _serializar(ej: Ejemplo) -> dict:
    return {
        "texto": ej.texto,
        "intencion": ej.intencion,
        "plantilla_id": ej.plantilla_id,
        "con_ruido": ej.con_ruido,
        "entidades_bio": ej.bio,
        "forma_logica": _fl_a_dict(ej.forma_logica),
        "sql_referencia": ej.sql_referencia,
        "respuesta_esperada": ej.respuesta_esperada,
    }


def main():
    ejemplos = construir_dataset()
    rnd = random.Random(SEED)
    train, val, test = particionar(ejemplos, rnd)

    con = conectar_solo_lectura()
    cat = cargar_catalogo(con)
    ood = construir_ood(cat, con)
    con.close()

    for nombre, subset in [("train", train), ("val", val), ("test", test), ("ood", ood)]:
        path = OUT_DIR / f"{nombre}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for ej in subset:
                f.write(json.dumps(_serializar(ej), ensure_ascii=False) + "\n")

    total = len(train) + len(val) + len(test)
    print(f"Dataset total (train+val+test): {total}")
    print(f"  train: {len(train)} ({100*len(train)/total:.1f}%)")
    print(f"  val:   {len(val)} ({100*len(val)/total:.1f}%)")
    print(f"  test:  {len(test)} ({100*len(test)/total:.1f}%)")
    print(f"  ood:   {len(ood)}")

    from collections import Counter
    print("\nDistribución por intención (total):")
    c = Counter(ej.intencion for ej in ejemplos)
    for intencion, n in c.most_common():
        print(f"  {intencion}: {n} ({100*n/total:.1f}%)")


if __name__ == "__main__":
    main()
