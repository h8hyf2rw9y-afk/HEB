# Registro de decisiones de arquitectura

Las decisiones marcadas como **Implementada** están fusionadas en `main` mediante un PR que aprobó el usuario. **Propuesta** significa que no hay implementación ni aprobación. Una propuesta de un agente no equivale a una aprobación del usuario.

| ID | Decisión | Motivo | Estado (2026-11-12) |
| --- | --- | --- | --- |
| ADR-001 | API con FastAPI + Pydantic | Fronteras tipadas entre UI, herramientas y RAG | Implementada (h8hyf2rw9y-afk/HEB#3, h8hyf2rw9y-afk/HEB#9) |
| ADR-002 | Interfaz en Streamlit | Demo funcional rápida en Python, utilizable por personas no técnicas | Implementada (h8hyf2rw9y-afk/HEB#7) |
| ADR-003 | DuckDB para datos estructurados | SQL determinista sobre CSV modestos, sin servidor | Motor implementado (h8hyf2rw9y-afk/HEB#3); registro de CSV y consultas **pendientes** (h8hyf2rw9y-afk/HEB#11) |
| ADR-004 | Inferencia local con Ollama | El usuario ya tiene Ollama local; costo monetario 0 | **Propuesta.** Ningún modelo integrado ni probado; ver ADR-010 |
| ADR-005 | Extracción local de PDF + recuperación indexada | Respuestas con evidencia sin mandar documentos completos a un modelo | Implementada (h8hyf2rw9y-afk/HEB#5, h8hyf2rw9y-afk/HEB#10) |
| ADR-006 | Coordinación de dos agentes por Issues, ramas y PR | Trazabilidad y sin ediciones concurrentes en archivos compartidos | En uso |
| ADR-007 | BM25 propio como recuperación por defecto; embeddings opcionales | Funciona sin Ollama ni GPU; determinista y medible | Implementada (h8hyf2rw9y-afk/HEB#5) |
| ADR-008 | Respuesta de políticas **extractiva**, sin LLM | Cada oración es verificable contra el PDF; el brief exige no inventar | Implementada (h8hyf2rw9y-afk/HEB#10) |
| ADR-009 | Precedencia entre documentos solo si la fuente la declara | Evitar resolver contradicciones por suposición, como "el más reciente gana" | Implementada (h8hyf2rw9y-afk/HEB#5, h8hyf2rw9y-afk/HEB#10) |
| ADR-010 | Elección de LLM y medición de costo en tokens | Requisito del brief (costo por interacción) | **Pendiente de decisión**; ver `docs/TOKEN_COSTS.md` |
| ADR-011 | Evaluación con un conjunto independiente escrito antes de los cambios | Evitar sobreajuste a las preguntas de desarrollo | Implementada (h8hyf2rw9y-afk/HEB#10) |
| ADR-012 | Umbral de abstención `min_coverage = 0.45` | En el conjunto independiente 0.45 = 0.50 y 0.40 es peor; sin evidencia para cambiarlo | Implementada (h8hyf2rw9y-afk/HEB#10) |

## Detalle

### ADR-007 — BM25 como recuperación por defecto
- **Contexto:** 4 PDF breves (29 páginas), español, una laptop sin GPU garantizada, y Ollama no disponible en todos los entornos (tampoco en el de desarrollo de los agentes).
- **Opciones:** embeddings densos (Ollama `bge-m3`); BM25 (`rank_bm25` u otra biblioteca); BM25 propio.
- **Decisión:** BM25 propio con normalización en español, y fusión híbrida RRF solo si hay embeddings.
- **Contras:** sensible al vocabulario (sinónimos, morfología); sin evidencia de cuánto mejoraría el modo híbrido, porque no se ha medido.

### ADR-008 — Respuestas extractivas
- **Contexto:** la gerente pide que no se inventen cifras y que cada política cite documento y sección.
- **Opciones:** LLM que redacta sobre pasajes; respuesta extractiva con reglas.
- **Decisión:** extractiva. Se copian oraciones, se citan y se separan en lo confirmado, la contradicción y lo no mencionado.
- **Contras:** respuestas menos naturales; algunas oraciones de más cuando el vocabulario se parece. Un LLM podría redactar encima de esto (ADR-010) sin reemplazar la recuperación ni la abstención.

### ADR-009 — Precedencia solo declarada por la fuente
- **Caso real:** FAQ P15 (15 días) frente a PRO-SAC-021 (30 días naturales). La FAQ dice que prevalece la política o el procedimiento, así que se resuelve a favor de PRO-SAC-021 y se cita esa frase.
- **Contras:** las fechas de vigencia se muestran, pero nunca deciden. Dos inconsistencias reales no se detectan (`docs/LIMITATIONS.md`).

### ADR-010 — LLM y costo (pendiente)
Se necesita decidir:
1. si se usa LLM y para qué (enrutar preguntas analíticas, redactar respuestas);
2. qué modelo y versión;
3. cómo medir los tokens.

`/chat` ya expone los campos de métricas. Sin esta decisión no se puede llenar la sección 4 de la plantilla de decisiones.

## Cómo añadir una decisión
Registra el contexto, las opciones consideradas, la decisión, sus contras, la fecha y la persona que la aprueba.
