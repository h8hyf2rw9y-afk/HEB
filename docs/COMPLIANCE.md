# Matriz de cumplimiento — ejercicio H-E-B

Fuente de los requisitos: [`candidatos/BRIEF.md`](../candidatos/BRIEF.md). Estado al **2026-11-12**, sobre `main` en `020c23c` (incluye DuckDB, h8hyf2rw9y-afk/HEB#11) y el PR de documentación h8hyf2rw9y-afk/HEB#14.

Leyenda:
- ✅ **Cumplido:** implementado, fusionado en `main` y verificado con la evidencia indicada.
- 🟡 **Parcial o con reservas:** se indica qué falta o qué hay que aclarar.
- ⏳ **Pendiente:** no implementado.
- 👤 **Del candidato:** solo la persona que entrega puede completarlo.

La evidencia se reproduce con:
```bash
python -m pytest tests -q
python scripts/verify_delivery.py
python scripts/demo_analytics.py
```

## A. Lo que el brief pide al sistema ("Lo que esperamos")

| # | Requisito del brief | Implementación | Evidencia | Estado |
| --- | --- | --- | --- | --- |
| A1 | Interfaz utilizable por una persona no técnica | Chat Streamlit en español (`frontend/app.py`): historial, estado, fuentes citadas y métricas; FastAPI (`app/api/main.py`) | Arranque verificado en entorno limpio y navegador real con preguntas de datos y de políticas (ver `docs/DEMO_SCRIPT.md`) | ✅ (no probada a mano en Windows) |
| A2a | Las **cifras salen de los datos**, no del modelo | DuckDB con SQL fijo y parametrizado sobre `candidatos/datos/*.csv` (`app/tools/analytics.py`, `app/data/catalog.py`); no hay modelo generativo | `tests/backend/test_analytics_reference.py` (cifras de la demo recalculadas con `csv`/`Decimal`); `scripts/verify_delivery.py` las recalcula de nuevo sin DuckDB; CI en Windows y Linux | ✅ Dentro del catálogo acotado de preguntas (ver `docs/LIMITATIONS.md` §1) |
| A2b | Las respuestas de **políticas** indican **documento y sección** | RAG `rag-v1` (`app/rag/`): citas `[archivo.pdf, p. N, sección: …]` | `tests/rag/test_candidatos.py::test_evaluation_battery` y `tests/rag/test_rag_v3_answers.py::test_holdout_battery_quality_floors`: 100 % de citas verificadas contra el PDF en 112 preguntas | ✅ |
| A3 | El sistema **reconoce lo que no puede responder** | Datos: abstención fuera del periodo, del corte o del catálogo (`app/tools/analytics.py`); políticas: abstención con motivo (`app/rag/service.py`) | Políticas: 0 abstenciones falsas y 93 % de abstenciones correctas en el conjunto independiente; datos: `tests/backend/test_analytics.py`, A4 en `scripts/verify_delivery.py` | 🟡 Cumplido en cada vía, pero el enrutamiento manda algunas preguntas de política a datos y responde con cifras ajenas (`docs/LIMITATIONS.md` §2) |
| A4 | Forma de saber **qué tan bien responde** | Políticas: 63 preguntas de desarrollo y 49 independientes; datos: pruebas con cifras calculadas de forma independiente; verificador de punta a punta | `tests/rag/fixtures/holdout_eval.json`, `tests/backend/test_analytics_reference.py`, `scripts/verify_delivery.py`, `app/rag/EVALUATION.md` | ✅ Políticas · 🟡 datos: solo los 3 casos de la demo, sin batería amplia |
| A5 | Reportar **cuánto cuesta en tokens** una interacción típica con el modelo elegido | Decisión: sin LLM generativo. `/chat` mide llamadas, tokens y latencia. | Medido: 0 llamadas LLM, 0 tokens y $0 por tokens en 8 preguntas típicas (`docs/TOKEN_COSTS.md`) | 🟡 Cumplido con la interpretación "modelo elegido = ninguno"; **conviene confirmarlo con la reclutadora** (`docs/TOKEN_COSTS.md`) |
| A6 | Preguntas en **español**, "como le preguntaría a su asistente" | Normalización en español en ambas vías | Todas las preguntas de prueba están en español natural | 🟡 Políticas ✅ · datos: catálogo acotado; formas no previstas se abstienen o se responden mal (`docs/LIMITATIONS.md` §1) |
| A7 | "Cuando no sepa, me lo diga, **no que me invente cifras**" | Sin generación: las cifras vienen de SQL y las políticas son citas textuales | `scripts/verify_delivery.py` (cifras recalculadas) y verificador de citas (`app/rag/evaluation.py`) | 🟡 No inventa cifras, pero ante una pregunta mal enrutada puede mostrar **cifras reales que no responden lo preguntado** (`docs/LIMITATIONS.md` §2) |

## B. Datos entregados ("Lo que te damos")

| # | Fuente | Uso actual | Evidencia | Estado |
| --- | --- | --- | --- | --- |
| B1 | `politicas/` (4 PDF) | 244 fragmentos (idénticos con PyMuPDF 1.28 y 2.1); tablas fila por fila; código, versión y vigencia | `scripts/verify_delivery.py`; `test_all_four_pdfs_are_indexed_with_document_facts` | ✅ |
| B2 | `candidatos/datos/ventas.csv` (jun–ago 2026) | Ventas netas, unidades y descuentos por mes, tienda, producto, categoría y canal; las devoluciones (negativas) restan | `tests/backend/test_analytics_reference.py` (A1) | ✅ |
| B3 | `candidatos/datos/inventario.csv` (corte 2026-09-01) | Productos en o por debajo del punto de reorden y agotados, siempre con fecha de corte; se abstiene para otras fechas | `tests/backend/test_analytics_reference.py` (A2) | ✅ |
| B4 | `candidatos/datos/tickets_mesa_servicio.csv` (285) | Conteos por prioridad, estado y tienda | `tests/backend/test_analytics_reference.py` (A3) | 🟡 Sin análisis del texto libre de los tickets |
| B5 | `candidatos/datos/catalogo_productos.csv`, `candidatos/datos/tiendas.csv` | Cruces de nombres de producto y tienda | `app/tools/analytics.py` | ✅ |
| B6 | `plantilla_decisiones.md` | Borrador con todo lo verificable | `docs/DECISIONES_BORRADOR.md` | 🟡 / 👤 Falta nombre, fecha, liga y la redacción propia |

## C. Entregables ("Entregables")

| # | Entregable | Estado | Evidencia | Estado |
| --- | --- | --- | --- | --- |
| C1 | Repositorio con **README que permita correrlo en < 10 minutos** | README con Windows (PowerShell) y macOS/Linux; `requirements.txt` raíz completo | Linux limpio: ≈ 1 min hasta la interfaz; Windows (CI): instalación 62 s y pruebas 22 s (`README.md`, «Tiempo de puesta en marcha») | 🟡 Falta cronometrar el arranque completo (API + Streamlit) en un Windows real |
| C2 | `plantilla_decisiones.md` llena | Borrador listo | `docs/DECISIONES_BORRADOR.md` | 👤 |
| C3 | **Video de 3 a 5 minutos** | Guion con salidas reales | `docs/DEMO_SCRIPT.md` | 👤 Pendiente de grabar (idealmente tras corregir h8hyf2rw9y-afk/HEB#12) |
| C4 | Envío por correo (repositorio o zip; liga si pesa > 20 MB) | — | — | 👤 |
| C5 | **Fecha límite:** 25 de septiembre de 2026, 23:59 (centro de México) | La fecha de trabajo de este entorno es 2026-11-12 | — | ⚠️ Confirmar el plazo vigente con la reclutadora |

## D. Requisitos técnicos del proyecto (`docs/REQUIREMENTS.md`, `AGENTS.md`, `CLAUDE.md`)

| # | Requisito | Evidencia | Estado |
| --- | --- | --- | --- |
| D1 | Separación UI / API / datos / RAG | `frontend/app.py`, `app/api/main.py`, `app/tools/analytics.py` + `app/data/catalog.py`, `app/rag/interface.py`; sin capa LLM, por decisión (ADR-010) | ✅ |
| D2 | Consultas deterministas, de solo lectura, sin SQL generado | SQL fijo con parámetros (`app/tools/analytics.py`, `app/data/engine.py`); `test_filter_parameterization` | ✅ |
| D3 | Validación de parámetros con Pydantic | `AggregateRequest` y `ChatRequest` (`app/api/main.py`), `PolicySearchRequest` (`app/rag/interface.py`); `test_interface_request_validation` | ✅ |
| D4 | Métricas por pregunta: tokens, latencia y llamadas | `/chat` devuelve `metrics` en ambas vías; la ruta no se expone explícitamente | 🟡 La ruta (datos o políticas) solo se deduce de las fuentes |
| D5 | Contradicciones con precedencia documentada | FAQ P15 frente a PRO-SAC-021, resuelta con la regla de la propia FAQ; `test_appliance_conflict_shows_both_sides_and_follows_the_faq_rule` | 🟡 1 detectada · 2 documentadas sin detección (`docs/LIMITATIONS.md` §4) |
| D6 | No subir secretos, `.venv`, índices ni datos privados | `.gitignore`; `test_generated_pdfs_and_index_are_ignored_by_git` | ✅ |
| D7 | Pruebas deterministas, multiplataforma | 218 pruebas pasan en local (1 omitida: Ollama); CI: 193 pasan en Windows y Linux (`tests/backend` + `tests/rag`); `test_answers_do_not_depend_on_hash_seed` | ✅ |
| D8 | Embeddings locales con Ollama (opcional) | Adaptador y RRF implementados | ⏳ **Nunca probado con un modelo real** |

## Qué falta para entregar
1. **Corregir h8hyf2rw9y-afk/HEB#12** (2 líneas en `app/rag/answering.py`; la corrección está probada). Si no, mencionarlo en el video o usar la pregunta alternativa.
2. **Decidir sobre los errores de enrutamiento** (`docs/LIMITATIONS.md` §2): corregirlos (Codex) o evitarlos en el video y documentarlos.
3. **Confirmar con la reclutadora** la fecha límite y si aceptan la solución sin LLM generativo para el requisito de costo en tokens.
4. **Cronometrar en un Windows real** los pasos 1–6 del README.
5. **Persona que entrega:** completar la plantilla a partir de `docs/DECISIONES_BORRADOR.md`, grabar el video con `docs/DEMO_SCRIPT.md` y enviar.
