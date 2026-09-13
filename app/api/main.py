"""
API minima (FastAPI + Pydantic, segun el stack tecnologico ya fijado en la Tabla 9
del Cap.3) que expone el asistente conversacional sobre HTTP.

Uso:
    uvicorn api.main:app --reload --port 8000

Prueba:
    curl -X POST localhost:8000/preguntar -H "Content-Type: application/json" \
         -d '{"pregunta": "¿Qué ejercicios tengo programados para el lunes?"}'
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.pipeline import Asistente  # noqa: E402

app = FastAPI(
    title="Asistente conversacional de entrenamiento (TFM)",
    description="Responde preguntas en lenguaje natural sobre la rutina programada "
                "y el historico de entrenamiento, apoyandose en NLU + acceso a datos "
                "estructurado sobre SQLite (ver Cap.3 de la Memoria).",
    version="0.1.0",
)

_asistente: Asistente | None = None


def _get_asistente() -> Asistente:
    global _asistente
    if _asistente is None:
        _asistente = Asistente()
    return _asistente


class PreguntaIn(BaseModel):
    pregunta: str


class RespuestaOut(BaseModel):
    pregunta: str
    intencion_detectada: str
    respuesta: str
    sql_ejecutado: str


@app.get("/salud")
def salud():
    return {"estado": "ok"}


@app.post("/preguntar", response_model=RespuestaOut)
def preguntar(payload: PreguntaIn) -> RespuestaOut:
    asistente = _get_asistente()
    r = asistente.preguntar(payload.pregunta)
    return RespuestaOut(
        pregunta=r.pregunta,
        intencion_detectada=r.intencion_detectada,
        respuesta=r.respuesta,
        sql_ejecutado=r.sql,
    )
