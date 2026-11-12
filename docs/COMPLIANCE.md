# Matriz de cumplimiento — ejercicio H-E-B

Fuente de los requisitos: [`candidatos/BRIEF.md`](../candidatos/BRIEF.md). Estado al **2026-11-12**, sobre `main` en `5c9fcc6` y el PR de documentación `docs/heb-final-delivery`.

Leyenda:
- ✅ **Cumplido:** implementado, fusionado en `main` y verificado con la evidencia indicada.
- 🟡 **Parcial:** una parte está cumplida y otra pendiente; se indica cuál.
- ⏳ **Pendiente:** no implementado o en un PR sin fusionar.
- 👤 **Del candidato:** solo la persona que entrega puede completarlo (nombre, video, envío).

La evidencia se puede reproducir con `python scripts/verify_delivery.py` y `python -m pytest tests -q`. **Ninguna fila analítica (DuckDB) se marca como cumplida hasta que Codex la entregue y se fusione.**

## A. Lo que el brief pide al sistema ("Lo que esperamos")

| # | Requisito del brief | Implementación existente | Evidencia | Estado |
| --- | --- | --- | --- | --- |
| A1 | Interfaz utilizable por una persona no técnica | Chat Streamlit en español (`frontend/app.py`): historial, estado, fuentes y métricas; FastAPI (`app/api/main.py`) | Arranque verificado en un entorno limpio (`/_stcore/health` = ok) y captura con una respuesta real (ver `docs/DEMO_SCRIPT.md`). La interfaz no se probó en Windows. | ✅ (con observaciones en h8hyf2rw9y-afk/HEB#13) |
| A2a | Las **cifras salen de los datos**, no del modelo | Catálogo DuckDB con consultas en lista blanca (`app/data/engine.py`, `app/tools/structured.py`). En `main` el catálogo está **vacío** (`build_catalog` devuelve `{}`) y `/chat` responde `insufficient_data` a preguntas de datos. | `tests/backend/test_structured.py` prueba el motor con datos de prueba; `verify_delivery.py` reporta `insufficient_data` en ventas, inventario y tickets. | ⏳ Pendiente de Codex (h8hyf2rw9y-afk/HEB#11, borrador "no fusionar") |
| A2b | Las respuestas de **políticas** indican **documento y sección** | RAG `rag-v1` (`app/rag/`): citas `[archivo.pdf, p. N, sección: …]` en `answer` y `evidence` con documento, página, sección y pasaje | `tests/rag/test_candidatos.py::test_evaluation_battery` y `tests/rag/test_rag_v3_answers.py::test_holdout_battery_quality_floors`: 100 % de citas verificables palabra por palabra contra el PDF en 112 preguntas | ✅ |
| A3 | El sistema **reconoce lo que no puede responder** | RAG: `insufficient_data` por cobertura, foco ausente, entidad ausente en una tabla o cifras de ventas; `/chat`: `insufficient_data` para datos estructurados mientras no exista DuckDB | Conjunto independiente: abstención correcta 0.93 y **0 abstenciones falsas**; desarrollo: 1.00 / 0 (ver [`../app/rag/EVALUATION.md`](../app/rag/EVALUATION.md)) | 🟡 Políticas ✅ · datos ⏳ (abstención genérica hasta que exista DuckDB) |
| A4 | Forma de saber **qué tan bien responde** (set de pruebas, métricas) | Dos conjuntos de evaluación de políticas: 63 de desarrollo y 49 independientes. CLI `python -m app.rag eval`. 188 pruebas automatizadas (166 de RAG y backend + 22 de documentación). | `tests/rag/fixtures/candidatos_eval.json`, `tests/rag/fixtures/holdout_eval.json`, [`../app/rag/EVALUATION.md`](../app/rag/EVALUATION.md) | 🟡 Políticas ✅ · preguntas analíticas sin set de evaluación ⏳ |
| A5 | Reportar **cuánto cuesta en tokens** una interacción típica con el modelo elegido | `/chat` mide `input_tokens`, `output_tokens`, `llm_calls`, `embedding_input_tokens` y latencia. **Hoy no hay ningún LLM en el sistema**: el RAG es extractivo y no existe `app/llm/`. | Medición real: 0 llamadas LLM y 0 tokens por pregunta de política (ver [`TOKEN_COSTS.md`](TOKEN_COSTS.md)) | ⏳ Hay que elegir modelo e integrarlo; ver la decisión pendiente en [`TOKEN_COSTS.md`](TOKEN_COSTS.md) |
| A6 | Respuestas en **español** "como le preguntaría a su asistente" | Normalización en español, respuestas y mensajes en español | Todas las preguntas de evaluación están en español natural (`tests/rag/fixtures/holdout_eval.json`) | ✅ Políticas · ⏳ consultas analíticas en lenguaje natural (h8hyf2rw9y-afk/HEB#11, pendiente) |
| A7 | "Cuando no sepa, me lo diga, **no que me invente cifras**" | Las respuestas de políticas solo copian oraciones de los PDF y no hay LLM que pueda inventar | Verificador de citas al 100 % (`app/rag/evaluation.py`, `CitationVerifier`) | ✅ Políticas · ⏳ datos |

## B. Datos entregados en el brief ("Lo que te damos")

| # | Fuente | Uso actual | Evidencia | Estado |
| --- | --- | --- | --- | --- |
| B1 | `politicas/` (4 PDF) | Indexados: 244 fragmentos, tablas fila por fila; código, versión y vigencia de cada documento | `scripts/verify_delivery.py`: «documentos=4 fragmentos=244»; `test_all_four_pdfs_are_indexed_with_document_facts` | ✅ |
| B2 | `datos/ventas.csv` (jun–ago 2026) | Sin registrar en `main` | — | ⏳ Codex (h8hyf2rw9y-afk/HEB#11) |
| B3 | `datos/inventario.csv` (corte 2026-09-01) | Sin registrar en `main` | — | ⏳ Codex |
| B4 | `datos/tickets_mesa_servicio.csv` (~300; 285 según el diccionario) | Sin registrar en `main` | — | ⏳ Codex |
| B5 | `datos/catalogo_productos.csv`, `datos/tiendas.csv` | Sin registrar en `main` (los necesitan los cruces de nombres) | — | ⏳ Codex |
| B6 | `plantilla_decisiones.md` | Borrador con lo que ya se puede afirmar y huecos marcados | [`DECISIONES_BORRADOR.md`](DECISIONES_BORRADOR.md) | 🟡 Borrador; la persona que entrega lo completa y lo copia a la plantilla |

## C. Entregables ("Entregables")

| # | Entregable | Estado actual | Evidencia | Estado |
| --- | --- | --- | --- | --- |
| C1 | Repositorio con **README que permita correrlo en < 10 minutos** | README reescrito con instalación para Windows (PowerShell) y macOS/Linux, y script de verificación | Medido en un **contenedor Linux** limpio: venv 4 s + instalación 44 s + índice 7–8 s + verificación 8 s ≈ **1 min**. **No medido en Windows** (ver [`../README.md`](../README.md)). | 🟡 Instrucciones listas; falta probar en Windows y que el `requirements.txt` raíz deje de estar vacío (h8hyf2rw9y-afk/HEB#13) |
| C2 | `plantilla_decisiones.md` llena: prioridades, descartes, dos semanas más y **costo por interacción** | Borrador preparado; el costo queda pendiente de A5 | [`DECISIONES_BORRADOR.md`](DECISIONES_BORRADOR.md) | 🟡 / 👤 |
| C3 | **Video de 3 a 5 minutos** con el sistema funcionando | Guion con tiempos, preguntas de política verificadas y espacios para las analíticas | [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) | 👤 Pendiente de grabar, después de las analíticas |
| C4 | Envío por correo (repositorio o zip; liga si pesa > 20 MB) | — | — | 👤 |
| C5 | **Fecha límite:** viernes 25 de septiembre de 2026, 23:59 (centro de México) | La fecha de trabajo de este entorno es 2026-11-12, **posterior** a la fecha límite del brief | — | ⚠️ Confirmar con la reclutadora si la fecha sigue vigente o se acordó otra |

## D. Requisitos técnicos propios del proyecto (`docs/REQUIREMENTS.md`, `AGENTS.md`, `CLAUDE.md`)

| # | Requisito | Evidencia | Estado |
| --- | --- | --- | --- |
| D1 | Separación UI / API / herramientas de datos / RAG / LLM | `frontend/`, `app/api/`, `app/tools/` + `app/data/`, `app/rag/`; `app/llm/` no existe todavía | 🟡 Sin capa LLM |
| D2 | Consultas deterministas y de solo lectura, sin SQL generado por el modelo | Identificadores en lista blanca y parámetros enlazados (`app/data/engine.py`); `test_filter_parameterization` | ✅ Motor · ⏳ catálogo real |
| D3 | Validación Pydantic de parámetros | `AggregateRequest` y `ChatRequest` (`app/api/main.py`), `PolicySearchRequest` (`app/rag/interface.py`); `test_interface_request_validation` | ✅ |
| D4 | Métricas por pregunta: tokens, latencia, llamadas y ruta | `/chat` devuelve `metrics`. Ruta: política o "no disponible". | 🟡 Sin LLM; la ruta solo es implícita |
| D5 | Contradicciones entre políticas con precedencia documentada | Conflicto FAQ P15 (15 días) frente a PRO-SAC-021 (30 días naturales), resuelto con la regla escrita en la propia FAQ; dos inconsistencias no detectadas, documentadas | `test_appliance_conflict_shows_both_sides_and_follows_the_faq_rule`; [`LIMITATIONS.md`](LIMITATIONS.md) | 🟡 1 detectada · 2 documentadas sin detección automática |
| D6 | No subir secretos, `.venv`, índices ni datos privados | `.gitignore` raíz (`/data/raw/`, `/data/rag_index/`…); los índices escriben su propio `.gitignore` | `git check-ignore data/rag_index/index.json` → ignorado; `test_generated_pdfs_and_index_are_ignored_by_git` | ✅ |
| D7 | Pruebas deterministas | 188 pruebas pasan, 1 omitida (Ollama); el RAG no depende de `PYTHONHASHSEED` | `python -m pytest tests -q`; `test_answers_do_not_depend_on_hash_seed` | ✅ |
| D8 | Embeddings locales con Ollama (opcional en la arquitectura) | Adaptador `OllamaEmbedder` y RRF implementados | **No probado con un modelo real**: no hay Ollama en el entorno de desarrollo | ⏳ Sin verificar |

## Qué necesita aportar Codex para cerrar la entrega
1. **Fusionar el registro de CSV y el catálogo (h8hyf2rw9y-afk/HEB#11)** con pruebas de integración sobre `candidatos/datos/`. Deben respetar el corte de inventario 2026-09-01, las devoluciones como unidades negativas y los SKU sustituidos o descontinuados.
2. **Consultas analíticas desde `/chat` en español**, con ruta explícita (DuckDB o RAG) y `evidence` de tipo `csv` con el periodo o la fecha de corte.
3. **Las 3–5 preguntas analíticas del video, con sus resultados reales.** Hay que rellenar los huecos de [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) solo con salidas reales, y añadir un set de evaluación analítico con respuestas esperadas calculadas de forma independiente.
4. **Decidir el LLM** (o justificar no usarlo) para poder medir y reportar el costo en tokens ([`TOKEN_COSTS.md`](TOKEN_COSTS.md)).
5. **Completar el `requirements.txt` raíz**, revisar el filtro `_POLICY_TERMS` y el render en Streamlit (h8hyf2rw9y-afk/HEB#13).
6. Confirmar que el README sigue siendo válido tras sus cambios, ejecutando `python scripts/verify_delivery.py`.
