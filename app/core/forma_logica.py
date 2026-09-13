"""
Forma logica: representacion intermedia y validada entre la intencion+entidades
detectadas por el NLU y la plantilla SQL que se ejecuta. Ver apartado 3.6/3.7 de la
Memoria: separar este paso permite evaluar por separado un fallo de comprension de
un fallo de traduccion a consulta.

Las 11 intenciones (10 originales del Cap.3 + consulta_programacion, anadida en esta
fase de implementacion para poder responder a la categoria "Entrenamiento actual"
sobre el plan semanal programado, que no tenia tabla propia en el diseno original).
"""
from __future__ import annotations

from dataclasses import dataclass, field

INTENCIONES = [
    "consulta_registro",
    "consulta_maximo",
    "consulta_volumen",
    "consulta_frecuencia",
    "consulta_progresion",
    "comparacion",
    "ultima_sesion",
    "consulta_catalogo",
    "consulta_programacion",
    "fuera_de_dominio",
    "ambigua",
]

METRICAS_VALIDAS = {
    "peso", "series", "repeticiones", "volumen", "tonelaje", "duracion", "rpe",
    # especificas de consulta_programacion (plan semanal, no historial):
    "grupo_muscular", "ejercicios_distintos",
}
AGREGACIONES_VALIDAS = {"maximo", "minimo", "media", "total", "ultimo", "conteo", "evolucion"}
DIAS_VALIDOS = {"Lunes", "Martes", "Miercoles"}


class ErrorFormaLogica(ValueError):
    """Se lanza cuando la forma logica no es valida o no es respondible (-> 'ambigua')."""


@dataclass
class Ambito:
    """Un ambito temporal/de filtro: usado como filtro principal y, en 'comparacion',
    tambien como segundo termino de la comparacion."""
    ejercicio: str | None = None
    grupo_muscular: str | None = None
    dia_semana: str | None = None
    semana: int | None = None            # semana puntual (1..13)
    semana_inicio: int | None = None      # rango [inicio, fin] inclusive
    semana_fin: int | None = None


@dataclass
class FormaLogica:
    intencion: str
    metrica: str | None = None
    agregacion: str | None = None
    ambito: Ambito = field(default_factory=Ambito)
    ordinal: str | None = None            # 'ultima' | 'penultima' | entero (n-esima)
    comparar_con: Ambito | None = None    # solo para 'comparacion'
    umbral: float | None = None
    umbral_operador: str | None = None    # '>' | '<' | '>=' | '<='
    top_n: int | None = None              # para rankings ("que ejercicios han aumentado mas")

    def validar(self) -> None:
        if self.intencion not in INTENCIONES:
            raise ErrorFormaLogica(f"Intencion desconocida: {self.intencion}")
        if self.intencion in ("fuera_de_dominio", "ambigua"):
            return
        if self.metrica is not None and self.metrica not in METRICAS_VALIDAS:
            raise ErrorFormaLogica(f"Metrica desconocida: {self.metrica}")
        if self.agregacion is not None and self.agregacion not in AGREGACIONES_VALIDAS:
            raise ErrorFormaLogica(f"Agregacion desconocida: {self.agregacion}")
        if self.ambito.dia_semana is not None and self.ambito.dia_semana not in DIAS_VALIDOS:
            raise ErrorFormaLogica(f"Dia no valido: {self.ambito.dia_semana}")
        if self.intencion == "comparacion" and self.comparar_con is None:
            raise ErrorFormaLogica("La intencion 'comparacion' requiere 'comparar_con'")
