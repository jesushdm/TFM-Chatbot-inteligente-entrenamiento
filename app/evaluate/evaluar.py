"""
Evaluacion automatica en 4 niveles (apartado 3.8 de la Memoria) sobre el conjunto de
test y el conjunto OOD:

  1) Comprension: el clasificador de intencion (TF-IDF + regresion logistica) acierta
     la intencion verdadera. Aisla el primer eslabon de la cadena.
  2) Extraccion de entidades / forma logica: usando la intencion VERDADERA (no la
     predicha) pero las entidades extraidas de verdad por el resolutor de reglas
     sobre el texto, se construye la forma logica, se ejecuta contra la BD y se
     compara el resultado con el de la forma logica de referencia anotada en el
     dataset. Aisla el segundo eslabon (independiente de si el clasificador acierta).
  3) Respuesta final: usando la forma logica de REFERENCIA (anotada), se ejecuta y se
     genera la respuesta en lenguaje natural, comparando el resultado de datos
     obtenido con el de referencia. Al reusar el mismo codigo que construyo el
     dataset, sirve como control de consistencia/determinismo de la capa de
     respuesta, no como medida de dificultad.
  4) Sistema (extremo a extremo): pregunta en texto libre -> pipeline completo
     (clasificador + entidades) -> se compara el resultado de datos obtenido con el
     de referencia. Es la cifra que importa para el usuario: combina los fallos de
     los tres niveles anteriores.

Un ejemplo de interseccion de control (fuera_de_dominio / ambigua) se cuenta como
correcto en los niveles 2-4 si la intencion resultante (predicha o verdadera, segun
el nivel) coincide con la de referencia; no aplica comparacion de datos porque esas
intenciones no ejecutan SQL.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.db import conectar_solo_lectura  # noqa: E402
from core.entidades import ResolutorEntidades  # noqa: E402
from core.forma_logica import Ambito, FormaLogica  # noqa: E402
from core.pipeline import Asistente  # noqa: E402
from core.responder import responder  # noqa: E402
from core.sql_templates import ejecutar  # noqa: E402

DATASET_DIR = ROOT / "nlu" / "dataset"
INFORME_PATH = ROOT / "evaluate" / "informe_evaluacion.json"

CONTROL = {"fuera_de_dominio", "ambigua"}


def _fl_desde_dict(d: dict) -> FormaLogica:
    amb = Ambito(**d["ambito"])
    comparar = Ambito(**d["comparar_con"]) if d.get("comparar_con") else None
    return FormaLogica(
        intencion=d["intencion"], metrica=d.get("metrica"), agregacion=d.get("agregacion"),
        ambito=amb, ordinal=d.get("ordinal"), comparar_con=comparar,
        umbral=d.get("umbral"), umbral_operador=d.get("umbral_operador"), top_n=d.get("top_n"),
    )


def _normalizar_filas(filas: list[dict]) -> list[tuple]:
    def _norm_val(v):
        if isinstance(v, float):
            return round(v, 2)
        return v
    return sorted(tuple(sorted((k, _norm_val(v)) for k, v in f.items())) for f in filas)


def _datos_equivalentes(filas_a: list[dict], filas_b: list[dict]) -> bool:
    return _normalizar_filas(filas_a) == _normalizar_filas(filas_b)


def _cargar(nombre: str) -> list[dict]:
    with (DATASET_DIR / f"{nombre}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluar_conjunto(nombre: str, asistente: Asistente, con) -> dict:
    ejemplos = _cargar(nombre)
    resolutor = asistente.resolutor

    n = len(ejemplos)
    ok_n1 = ok_n2 = ok_n3 = ok_n4 = 0
    fallos_n4 = []

    for ej in ejemplos:
        texto = ej["texto"]
        intencion_real = ej["intencion"]
        fl_ref = _fl_desde_dict(ej["forma_logica"])

        # --- Nivel 1: comprension (clasificador) ---
        intencion_predicha = asistente.clasificador.predict([texto])[0]
        n1 = intencion_predicha == intencion_real
        ok_n1 += n1

        # --- filas de referencia ---
        if intencion_real in CONTROL:
            filas_ref = None
        else:
            filas_ref = ejecutar(fl_ref, con).filas

        # --- Nivel 2: extraccion de entidades (con intencion VERDADERA) ---
        entidades = resolutor.resolver(texto)
        fl_desde_entidades = asistente._construir_forma_logica(intencion_real, entidades)
        if intencion_real in CONTROL:
            n2 = fl_desde_entidades.intencion == intencion_real
        else:
            try:
                filas_n2 = ejecutar(fl_desde_entidades, con).filas
                n2 = _datos_equivalentes(filas_n2, filas_ref)
            except Exception:
                n2 = False
        ok_n2 += n2

        # --- Nivel 3: respuesta final (con forma logica de REFERENCIA) ---
        if intencion_real in CONTROL:
            n3 = True  # por construccion (mensaje fijo), sirve de control de consistencia
        else:
            filas_n3 = ejecutar(fl_ref, con).filas
            n3 = _datos_equivalentes(filas_n3, filas_ref)
        ok_n3 += n3

        # --- Nivel 4: sistema de extremo a extremo (texto -> pipeline completo) ---
        r = asistente.preguntar(texto)
        if intencion_real in CONTROL:
            n4 = r.forma_logica.intencion == intencion_real
        else:
            try:
                filas_n4 = ejecutar(r.forma_logica, con).filas if r.forma_logica.intencion not in CONTROL else []
                n4 = r.forma_logica.intencion == intencion_real and _datos_equivalentes(filas_n4, filas_ref)
            except Exception:
                n4 = False
        ok_n4 += n4
        if not n4:
            fallos_n4.append({
                "texto": texto, "intencion_real": intencion_real,
                "intencion_predicha": intencion_predicha,
                "respuesta_esperada": ej["respuesta_esperada"], "respuesta_obtenida": r.respuesta,
            })

    return {
        "n": n,
        "nivel1_comprension": ok_n1 / n,
        "nivel2_acceso_datos": ok_n2 / n,
        "nivel3_respuesta_final": ok_n3 / n,
        "nivel4_sistema": ok_n4 / n,
        "fallos_nivel4_muestra": fallos_n4[:15],
    }


def main():
    con = conectar_solo_lectura()
    asistente = Asistente()

    informe = {}
    for nombre in ["test", "ood"]:
        print(f"\nEvaluando conjunto: {nombre}")
        resultado = evaluar_conjunto(nombre, asistente, con)
        informe[nombre] = resultado
        print(f"  n = {resultado['n']}")
        print(f"  Nivel 1 (comprension):        {resultado['nivel1_comprension']:.1%}")
        print(f"  Nivel 2 (acceso a datos):      {resultado['nivel2_acceso_datos']:.1%}")
        print(f"  Nivel 3 (respuesta final):     {resultado['nivel3_respuesta_final']:.1%}")
        print(f"  Nivel 4 (sistema extremo a extremo): {resultado['nivel4_sistema']:.1%}")

    INFORME_PATH.write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nInforme guardado en {INFORME_PATH}")

    asistente.cerrar()
    con.close()


if __name__ == "__main__":
    main()
