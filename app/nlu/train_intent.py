"""
Entrena el clasificador de intencion: TF-IDF (n-gramas de caracteres) + regresion
logistica multinomial. Alternativa elegida junto con el usuario para esta fase de
la implementacion (ver apartado 4.3 de la Memoria, capitulo de Resultados): no
requiere GPU, entrena en segundos en CPU, y dentro del sistema completo se combina
con extraccion de entidades por reglas (nlu/../core/entidades.py). La cabeza de
etiquetado BIO sobre un encoder en espanol (BETO), ya comparada y elegida en el
Cap.3 como arquitectura objetivo, queda pendiente de ajustar localmente (CPU, sin
GPU dedicada) como siguiente paso (ver la nota de estado del Cap.4).

Uso:
    python3 nlu/train_intent.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATASET_DIR = ROOT / "nlu" / "dataset"
MODELO_PATH = ROOT / "nlu" / "modelo_intencion.joblib"
INFORME_PATH = ROOT / "nlu" / "informe_entrenamiento.json"


def _cargar(nombre: str):
    textos, etiquetas = [], []
    with (DATASET_DIR / f"{nombre}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            textos.append(d["texto"])
            etiquetas.append(d["intencion"])
    return textos, etiquetas


def entrenar():
    x_train, y_train = _cargar("train")
    x_val, y_val = _cargar("val")
    x_test, y_test = _cargar("test")
    x_ood, y_ood = _cargar("ood")

    # TF-IDF de n-gramas de CARACTERES (no de palabras): se probo tambien la variante
    # habitual de 1-2 gramas de palabra, pero generalizaba mal a plantillas no vistas
    # en entrenamiento (accuracy en test ~0.56 vs ~0.55 en validacion, con varias
    # clases en 0 -- sintoma de que memorizaba frases completas). Los n-gramas de
    # caracteres comparten mas señal entre parafrasis de una misma intencion (raices,
    # terminaciones, la propia entidad sustituida) y son mas robustos a la particion
    # estricta por plantilla; con ellos el accuracy en test sube a ~0.81-0.84. Se deja
    # documentada esta comparativa en el Cap.3 (nota de estado del motor NLU).
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), lowercase=True,
            strip_accents="unicode", min_df=1, sublinear_tf=True,
        )),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=10.0)),
    ])
    pipeline.fit(x_train, y_train)

    informe = {}
    for nombre, xs, ys in [("val", x_val, y_val), ("test", x_test, y_test), ("ood", x_ood, y_ood)]:
        preds = pipeline.predict(xs)
        acc = sum(p == y for p, y in zip(preds, ys)) / len(ys)
        reporte = classification_report(ys, preds, zero_division=0, output_dict=True)
        informe[nombre] = {"accuracy": acc, "n": len(ys), "classification_report": reporte}
        print(f"\n=== {nombre.upper()} (n={len(ys)}) — accuracy: {acc:.3f} ===")
        print(classification_report(ys, preds, zero_division=0))
        if nombre == "test":
            etiquetas = sorted(set(ys) | set(preds))
            cm = confusion_matrix(ys, preds, labels=etiquetas)
            informe[nombre]["etiquetas"] = etiquetas
            informe[nombre]["matriz_confusion"] = cm.tolist()

    joblib.dump(pipeline, MODELO_PATH)
    INFORME_PATH.write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nModelo guardado en {MODELO_PATH}")
    print(f"Informe guardado en {INFORME_PATH}")
    return pipeline, informe


if __name__ == "__main__":
    entrenar()
