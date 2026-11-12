# Arquitectura

Estado real al 2026-11-12 (`main` en `5c9fcc6`). Lo marcado con ⏳ no está implementado en `main`.

```text
Navegador ── Streamlit (frontend/app.py, :8501)
                │  POST /chat {question}            GET /health · /health/rag
                ▼
           FastAPI (app/api/main.py, :8000) ── validación Pydantic
                │
                ├─ ¿pregunta de política? (filtro _POLICY_TERMS)
                │      sí ─► RAG rag-v1 (app/rag) ─► respuesta citada + evidencia + conflictos + métricas
                │      no ─► "consulta de datos estructurados aún no disponible" (insufficient_data)
                │
                ├─ ⏳ consultas analíticas ─► herramientas en lista blanca (app/tools) ─► DuckDB (app/data)
                │                               sobre candidatos/datos/*.csv  (h8hyf2rw9y-afk/HEB#11)
                └─ ⏳ capa LLM (app/llm) para enrutar y redactar; hoy no existe

Indexación (offline, una vez):  4 PDF ─PyMuPDF─► bloques + filas de tabla ─► fragmentos ─► data/rag_index/index.json
```

## Componentes

| Capa | Ubicación | Responsable | Estado |
| --- | --- | --- | --- |
| Interfaz | `frontend/app.py` (Streamlit) | Compartido (implementado por Codex, h8hyf2rw9y-afk/HEB#7) | ✅ Chat, estado, fuentes y métricas |
| API | `app/api/main.py` (FastAPI) | Codex | ✅ `/health`, `/health/rag`, `/chat`, `/query/aggregate` |
| Datos estructurados | `app/data/`, `app/tools/` (DuckDB) | Codex | 🟡 Motor seguro en lista blanca; catálogo vacío en `main`; registro de CSV en h8hyf2rw9y-afk/HEB#11 |
| Políticas (RAG) | `app/rag/` | Claude Code | ✅ Contrato `rag-v1` (`docs/CONTRACTS.md`, `app/rag/interface.py`) |
| LLM | `app/llm/` | Compartido | ⏳ No existe |

## RAG de políticas (cómo funciona)

### 1. Extracción (`app/rag/extraction.py`, PyMuPDF)
- **Texto con estructura:** se extrae con número de página y sección. Las secciones se detectan por negrita o tamaño de letra y numeración, incluidos los encabezados de varias líneas como "P18. …".
- **Encabezados y pies de página:** se eliminan los que se repiten en todas las páginas ("H-E-B México — Documento interno…", "Página N").
- **Tablas:** se leen con `find_tables`. Cada fila se convierte en una oración citable, "Encabezado: valor | …", para que "Electrodomésticos … | Plazo: 30 días naturales" no se parta.
- **Datos de cada documento:** código, versión y fecha de vigencia, tomados de la tabla de portada. También se registran los códigos de otros documentos que cita cada PDF y las frases de precedencia ("prevalece siempre la política o el procedimiento…").

### 2. Fragmentación (`chunking.py`)
- La prosa se agrupa por página y sección, en ventanas de 160 palabras con solape de 40.
- Cada fila de tabla es un fragmento propio.
- Ningún fragmento cruza página ni sección, para que la cita sea exacta.

Con los 4 PDF del ejercicio salen 244 fragmentos; el índice tarda unos 7–8 s en construirse.

### 3. Recuperación (`index.py`, `bm25.py`)
- **BM25** propio, sin dependencias, con normalización en español: acentos, plurales, palabras vacías y raíz de 6 letras.
- **Expansión basada en el corpus:** solo usa listas explícitas de las secciones "Definiciones"; por ejemplo, *licuadora → electrodomésticos y electrónicos menores*. Las coincidencias suaves (por prefijo o expansión) valen la mitad.
- **Híbrido opcional:** con embeddings de Ollama se fusiona con RRF. **No probado con un modelo real** (ver `LIMITATIONS.md`).

### 4. Selección y respuesta (`answering.py`, `service.py`)
- **Abstención (`insufficient_data`) con motivo**, en cualquiera de estos casos:
  - ningún pasaje cubre ≥ 45 % del peso IDF de la pregunta (`HEB_RAG_MIN_COVERAGE`);
  - el atributo preguntado ("¿Cuál es el **precio**…?") no aparece en la evidencia;
  - la tabla consultada no contiene la entidad preguntada (y se indica qué contiene);
  - se piden cifras de ventas.
- **Orden de la evidencia:** se ordena por qué tan directamente responde: cobertura, título de la sección y fila de tabla con la entidad y el atributo preguntados.
- **Texto de la respuesta:** solo oraciones **copiadas** del PDF, con su cita `[archivo, p. N, sección]`, más sus restricciones ("solo", "no", "exclusivamente"…). Los pasajes secundarios solo entran si corroboran desde otro documento o forman parte de un conflicto.
- **Contradicciones:** se comparan cifras y permitido/prohibido. Se muestran las dos partes y solo se marca cuál prevalece si **la fuente lo dice**.
- **Métricas:** latencia, modo de recuperación y embeddings. Las llamadas y tokens de LLM son siempre 0, porque no hay LLM.

### 5. Evaluación
Dos conjuntos: 63 preguntas de desarrollo y 49 independientes, escritas antes de la última ronda de cambios. Se ejecutan con `python -m app.rag eval` y con pruebas automatizadas, y cada cita se verifica palabra por palabra contra la página del PDF. Resultados en [`../app/rag/EVALUATION.md`](../app/rag/EVALUATION.md).

## Por qué BM25 y sin LLM (hoy)
- **Verificable:** cada oración de la respuesta existe en el PDF citado; no hay texto generado que pueda inventar cifras. Es lo que pide el brief: "que no me invente cifras".
- **Sin dependencias ni costo:** funciona sin Ollama ni GPU y responde en milisegundos.
- **Coste aceptado:** depende del vocabulario. Falla con sinónimos y morfología ("vender" frente a "venta") y produce respuestas extractivas, no redactadas. Ver [`LIMITATIONS.md`](LIMITATIONS.md).

Si se añade un LLM, el diseño previsto es que **redacte sobre la evidencia** (`app.rag.prompts.build_grounded_prompt`), sin sustituir la recuperación ni las reglas de abstención, y que cada llamada reporte sus tokens (ver [`TOKEN_COSTS.md`](TOKEN_COSTS.md)).

## Límites de arquitectura
- Los CSV se consultan con DuckDB; no se pasan tablas crudas a ningún modelo.
- El modelo, si se incorpora, no puede generar SQL ejecutable: elige herramientas con parámetros validados.
- **Despliegue:** solo local. Una API alojada en la nube no puede alcanzar el Ollama de una laptop (`localhost`).
- **Datos privados e índices** (`data/raw/`, `data/rag_index/`…) están en `.gitignore`. Los índices además escriben su propio `.gitignore`.

## Propiedad y coordinación
- Codex: `app/api/`, `app/data/`, `app/tools/`.
- Claude Code: `app/rag/`.
- Compartido con Issue acordado: `frontend/`, `app/llm/`, contratos, dependencias, CI y documentación de entrega.

Coordinación en GitHub Issues y Pull Requests; el usuario aprueba los merges. Ver `docs/HANDOFF.md` y `docs/CONTRACTS.md`.
