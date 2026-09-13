# Asistente conversacional de entrenamiento (TFM)

Sistema de preguntas y respuestas en lenguaje natural sobre una base de datos de
entrenamiento de fuerza. Implementa el Capítulo 3 (Metodología) y el Capítulo 4
(Resultados) de la Memoria: clasificador de intención (TF-IDF + regresión
logística), resolución de entidades por reglas, plantillas SQL parametrizadas
sobre SQLite y verbalización por plantillas. No requiere GPU ni servicios en la
nube.

## 1. Requisitos

- Python 3.11 o 3.12.
- Sin GPU: todo lo que hay en este repo (base de datos, clasificador ya
  entrenado, evaluación) se ejecuta en CPU.

## 2. Instalación

Desde la carpeta `app/` (raíz de este proyecto):

```bash
python3 -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Ya viene todo generado, no hace falta regenerar nada

El repositorio incluye la base de datos ya construida (`db/tfm.db`), el corpus
anotado (`nlu/dataset/*.jsonl`), el clasificador ya entrenado
(`nlu/modelo_intencion.joblib`) y los informes de entrenamiento y evaluación
(`nlu/informe_entrenamiento.json`, `evaluate/informe_evaluacion.json`). Para
probar el sistema basta con instalar las dependencias (paso 2) y seguir el
paso 4.

## 4. Uso rápido

### 4.1. Demo por consola

Ejecuta el pipeline completo contra las 24 preguntas de ejemplo del dominio:

```bash
python3 core/pipeline.py
```

Para hacer una pregunta suelta desde Python:

```python
from core.pipeline import Asistente

asistente = Asistente()
r = asistente.preguntar("¿Qué ejercicios tengo programados para el lunes?")
print(r.respuesta)
asistente.cerrar()
```

### 4.2. API REST

```bash
uvicorn api.main:app --reload --port 8000
```

Comprobación de estado:

```bash
curl localhost:8000/salud
```

Pregunta:

```bash
curl -X POST localhost:8000/preguntar \
     -H "Content-Type: application/json" \
     -d '{"pregunta": "¿Cuánto he progresado en press banca durante las últimas 4 semanas?"}'
```

La respuesta incluye la intención detectada, la respuesta en lenguaje natural
y el SQL exacto que se ejecutó (útil para depurar o para la defensa del TFM).

## 5. Regenerar desde cero

Solo hace falta si cambias los ficheros fuente (`data/source/*.md`), los
parámetros del generador, el propio dataset de preguntas o quieres reentrenar
el clasificador. Ejecutar en este orden desde `app/`:

```bash
# 1. Base de datos sintética a partir de Rutina_entrenamiento.md y Registro_historico.md
python3 db/build_db.py

# 2. Corpus anotado de preguntas (plantillas + ruido lingüístico por reglas)
python3 nlu/dataset_preguntas.py

# 3. Clasificador de intención (TF-IDF + regresión logística)
python3 nlu/train_intent.py

# 4. Evaluación en 4 niveles sobre los conjuntos de prueba y fuera de distribución
python3 evaluate/evaluar.py
```

`db/build_db.py` llama internamente a `data/generar_datos.py`; no hace falta
ejecutarlo aparte.

## 6. Estructura del proyecto

```
app/
├── api/            API REST (FastAPI): endpoints /salud y /preguntar
├── core/           Pipeline: BD, entidades, forma lógica, plantillas SQL, respuesta
│   └── pipeline.py   Asistente — punto de entrada único usado por la API, el CLI y la evaluación
├── data/
│   ├── source/       Ficheros fuente reales del usuario (rutina y registro histórico)
│   └── generar_datos.py   Generador sintético (13 semanas, semilla 42)
├── db/
│   ├── schema.sql    Esquema de 8 tablas (incluye rutina_ejercicio, el plan semanal)
│   ├── build_db.py   Construye db/tfm.db
│   └── tfm.db        Base de datos SQLite ya generada
├── nlu/
│   ├── dataset_preguntas.py   Genera el corpus anotado (nlu/dataset/*.jsonl)
│   ├── train_intent.py        Entrena el clasificador de intención
│   └── modelo_intencion.joblib
├── evaluate/
│   └── evaluar.py    Evaluación en 4 niveles (comprensión, acceso a datos, respuesta, sistema)
├── tests/            Vacío por ahora; sin pruebas unitarias en esta fase
└── requirements.txt
```

## 7. Qué falta y qué queda fuera de alcance (ver la nota de estado del Capítulo 4 de la Memoria)

Genuinamente pendiente:

- **Corpus**: 638 preguntas + 7 fuera de distribución, por debajo del objetivo
  de 1.200-1.500 (apartado 3.5). Ampliarlo es añadir plantillas en
  `nlu/dataset_preguntas.py` y repetir los pasos 3 y 4 de la sección 5.
- **tests/**: sin pruebas unitarias todavía.

Decidido como fuera del alcance entregado en esta fase (no son tareas
pendientes a tu cargo, ver Cap. 4 de la Memoria):

- **Codificador preentrenado ajustado** (BETO / RoBERTa-base-BNE, arquitectura
  objetivo del apartado 3.6). El clasificador que sí funciona hoy, y el que se
  entrega en esta fase, es el TF-IDF + regresión logística.
- **Verbalización opcional con modelo local** (apartado 3.7) y **paráfrasis del
  corpus** (apartado 3.5.2): no implementadas; si en el futuro se retoman, el
  modelo previsto es gemma-4-e4b servido con LM Studio.
