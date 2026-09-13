-- Esquema de la base de datos SQLite del asistente conversacional de entrenamiento.
-- Corresponde al modelo de datos descrito en el Capitulo 3 (Metodologia) de la Memoria del TFM,
-- ampliado con la tabla rutina_ejercicio para representar el PLAN semanal (lo programado),
-- separado de sesion/serie (lo realmente realizado).
--
-- Se ejecuta de forma idempotente (build_db.py la invoca sobre una base de datos nueva).

PRAGMA foreign_keys = ON;

-- Datos estables de la persona que entrena (un unico usuario sintetico en esta version).
CREATE TABLE IF NOT EXISTS usuario (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre              TEXT    NOT NULL,
    fecha_nacimiento    TEXT    NOT NULL,
    sexo                TEXT    NOT NULL,
    altura_cm           REAL    NOT NULL,
    nivel               TEXT    NOT NULL
);

-- Serie temporal de peso corporal y perimetros (opcional; no usada por las preguntas actuales,
-- se mantiene por coherencia con el modelo de datos ya documentado en la Memoria).
CREATE TABLE IF NOT EXISTS medida_corporal (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    usuario_id      INTEGER NOT NULL,
    fecha           TEXT    NOT NULL,
    peso_kg         REAL    NOT NULL,
    perimetros_json TEXT,
    FOREIGN KEY (usuario_id) REFERENCES usuario(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_medida_usuario ON medida_corporal(usuario_id);

-- Catalogo de ejercicios. grupo_muscular es la clasificacion estandar usada para responder
-- preguntas de tipo "que grupos musculares entreno cada dia" (no vinia en los archivos del
-- usuario, se ha asignado siguiendo un criterio anatomico habitual: ver nota en informe).
CREATE TABLE IF NOT EXISTS ejercicio (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre          TEXT    NOT NULL UNIQUE,
    grupo_muscular  TEXT    NOT NULL,
    equipamiento    TEXT    NOT NULL,
    unilateral      INTEGER NOT NULL DEFAULT 0,
    es_isometrico   INTEGER NOT NULL DEFAULT 0  -- 1 = no se registra peso (p.ej. plancha)
);

-- Denominaciones alternativas de cada ejercicio, usadas por la resolucion de entidades
-- (fuzzy matching) para traducir el vocabulario del usuario al ejercicio_id canonico.
CREATE TABLE IF NOT EXISTS alias_ejercicio (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ejercicio_id    INTEGER NOT NULL,
    alias           TEXT    NOT NULL,
    FOREIGN KEY (ejercicio_id) REFERENCES ejercicio(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_alias_ejercicio ON alias_ejercicio(ejercicio_id);

-- Bloque de entrenamiento o mesociclo (metadatos: nombre, fechas, objetivo).
CREATE TABLE IF NOT EXISTS rutina (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre          TEXT    NOT NULL,
    fecha_inicio    TEXT    NOT NULL,
    fecha_fin       TEXT    NOT NULL,
    objetivo        TEXT    NOT NULL
);

-- NUEVA (no estaba en el modelo de datos original del Cap.3): plan semanal programado.
-- Representa lo que TOCA cada dia de la semana segun la rutina vigente (series, rango de
-- repeticiones y RIR objetivo), independientemente de si esa sesion ya se ha realizado o no.
-- Es la tabla que responde a las preguntas de la categoria "Entrenamiento actual" del usuario
-- (p.ej. "que ejercicios tengo programados para el lunes").
CREATE TABLE IF NOT EXISTS rutina_ejercicio (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    rutina_id           INTEGER NOT NULL,
    dia_semana          TEXT    NOT NULL CHECK (dia_semana IN
                            ('Lunes','Martes','Miercoles','Jueves','Viernes','Sabado','Domingo')),
    ejercicio_id        INTEGER NOT NULL,
    orden               INTEGER NOT NULL,
    series_objetivo     INTEGER NOT NULL,
    rep_min             INTEGER NOT NULL,
    rep_max             INTEGER NOT NULL,
    rir_objetivo        INTEGER NOT NULL,
    FOREIGN KEY (rutina_id) REFERENCES rutina(id) ON DELETE CASCADE,
    FOREIGN KEY (ejercicio_id) REFERENCES ejercicio(id)
);

CREATE INDEX IF NOT EXISTS idx_plan_rutina ON rutina_ejercicio(rutina_id, dia_semana);

-- Sesion de entrenamiento REALIZADA (lo historico).
CREATE TABLE IF NOT EXISTS sesion (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    usuario_id      INTEGER NOT NULL,
    rutina_id       INTEGER NOT NULL,
    fecha           TEXT    NOT NULL,
    semana          INTEGER NOT NULL,   -- numero de semana del bloque (1..N), util para agregaciones
    dia_semana      TEXT    NOT NULL,
    duracion_min    INTEGER,
    rpe_sesion      REAL,
    notas           TEXT,
    FOREIGN KEY (usuario_id) REFERENCES usuario(id) ON DELETE CASCADE,
    FOREIGN KEY (rutina_id) REFERENCES rutina(id)
);

CREATE INDEX IF NOT EXISTS idx_sesion_usuario_fecha ON sesion(usuario_id, fecha);
CREATE INDEX IF NOT EXISTS idx_sesion_semana ON sesion(semana);

-- Unidad minima de registro: una serie de un ejercicio dentro de una sesion realizada.
CREATE TABLE IF NOT EXISTS serie (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    sesion_id           INTEGER NOT NULL,
    ejercicio_id        INTEGER NOT NULL,
    orden_ejercicio     INTEGER NOT NULL,
    numero_serie        INTEGER NOT NULL,
    repeticiones        INTEGER NOT NULL,
    peso_kg             REAL,        -- NULL para ejercicios isometricos (p.ej. plancha)
    rir                 INTEGER,
    tipo_serie          TEXT    NOT NULL DEFAULT 'efectiva',
    FOREIGN KEY (sesion_id) REFERENCES sesion(id) ON DELETE CASCADE,
    FOREIGN KEY (ejercicio_id) REFERENCES ejercicio(id)
);

CREATE INDEX IF NOT EXISTS idx_serie_sesion ON serie(sesion_id);
CREATE INDEX IF NOT EXISTS idx_serie_ejercicio ON serie(ejercicio_id);
