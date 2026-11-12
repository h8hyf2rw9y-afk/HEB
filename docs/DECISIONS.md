# Registro de decisiones de arquitectura

Las decisiones marcadas como **Implementada** están fusionadas en `main` mediante un PR que aprobó el usuario. **Propuesta** significa que no hay implementación ni aprobación. Una propuesta de un agente no equivale a una aprobación del usuario.

| ID | Decisión | Motivo | Estado (2026-11-12) |
| --- | --- | --- | --- |
| ADR-001 | API con FastAPI + Pydantic | Fronteras tipadas entre UI, herramientas y RAG | Implementada (h8hyf2rw9y-afk/HEB#3, h8hyf2rw9y-afk/HEB#9) |
| ADR-002 | Interfaz en Streamlit | Demo funcional rápida en Python, utilizable por personas no técnicas | Implementada (h8hyf2rw9y-afk/HEB#7) |
| ADR-003 | DuckDB para datos estructurados | SQL determinista sobre CSV modestos, sin servidor | Implementada (h8hyf2rw9y-afk/HEB#3, h8hyf2rw9y-afk/HEB#11) |
| ADR-004 | Inferencia local con Ollama | El usuario ya tiene Ollama local; costo monetario 0 | **No adoptada** para generar respuestas (ADR-010). Embeddings opcionales implementados, nunca probados con un modelo real. |
| ADR-005 | Extracción local de PDF + recuperación indexada | Respuestas con evidencia sin mandar documentos completos a un modelo | Implementada (h8hyf2rw9y-afk/HEB#5, h8hyf2rw9y-afk/HEB#10) |
| ADR-006 | Coordinación de dos agentes por Issues, ramas y PR | Trazabilidad y sin ediciones concurrentes en archivos compartidos | En uso |
| ADR-007 | BM25 propio como recuperación por defecto; embeddings opcionales | Funciona sin Ollama ni GPU; determinista y medible | Implementada (h8hyf2rw9y-afk/HEB#5) |
| ADR-008 | Respuesta de políticas **extractiva**, sin LLM | Cada oración es verificable contra el PDF; el brief exige no inventar | Implementada (h8hyf2rw9y-afk/HEB#10) |
| ADR-009 | Precedencia entre documentos solo si la fuente la declara | Evitar resolver contradicciones por suposición, como "el más reciente gana" | Implementada (h8hyf2rw9y-afk/HEB#5, h8hyf2rw9y-afk/HEB#10) |
| ADR-010 | **Sin LLM generativo**: cifras con SQL fijo y políticas con citas textuales; costo medido de 0 tokens | Verificabilidad, costo y reproducibilidad | **Decidida** (indicación del usuario, 2026-11-12); falta confirmar con la reclutadora que satisface el requisito de costo (`docs/TOKEN_COSTS.md`) |
| ADR-011 | Evaluación con un conjunto independiente escrito antes de los cambios | Evitar sobreajuste a las preguntas de desarrollo | Implementada (h8hyf2rw9y-afk/HEB#10) |
| ADR-012 | Umbral de abstención `min_coverage = 0.45` | En el conjunto independiente 0.45 = 0.50 y 0.40 es peor; sin evidencia para cambiarlo | Implementada (h8hyf2rw9y-afk/HEB#10) |
| ADR-013 | Enrutamiento por reglas léxicas: lo que parece de datos va a DuckDB y el resto al RAG | Sin modelo, determinista y explicable | Implementada (h8hyf2rw9y-afk/HEB#11); 3 de 112 preguntas de política mal enrutadas (`docs/LIMITATIONS.md` §2) |
| ADR-014 | Catálogo acotado de intenciones con SQL fijo y parametrizado (sin SQL generado) | Ninguna cifra sin consulta revisada; abstención fuera del catálogo | Implementada (h8hyf2rw9y-afk/HEB#11) |

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

### ADR-010 — Sin LLM generativo
- **Contexto:** el brief pide reportar el costo en tokens "con el modelo que hayas elegido", y evalúa "si sabes lo que cuesta". La gerente pide que no se inventen cifras.
- **Opciones:**
  - LLM que enrute y redacte (local con Ollama o por API);
  - LLM que solo redacte sobre la evidencia;
  - sin LLM.
- **Decisión (usuario, 2026-11-12):** sin LLM generativo. Las cifras salen de SQL fijo y las políticas son citas textuales. Costo medido: 0 llamadas, 0 tokens y $0 por tokens (`docs/TOKEN_COSTS.md`).
- **Contras:**
  - lenguaje acotado (catálogo de intenciones, reglas de enrutamiento);
  - respuestas no redactadas;
  - riesgo de que el evaluador espere ver un modelo. Se recomienda confirmarlo con la reclutadora.

### ADR-013 — Enrutamiento por reglas
- **Decisión:** `looks_like_data` (`app/tools/analytics.py`) envía a DuckDB las preguntas con vocabulario de datos e intención analítica, y el resto al RAG. Sustituyó el filtro anterior, que dejaba fuera preguntas de política válidas como "¿Qué es SIGMA?".
- **Contras:** al pasar las 112 preguntas de política de las baterías del RAG, 3 se desvían a datos y reciben cifras que no responden lo preguntado. Detalle en `docs/LIMITATIONS.md` §2.

## Cómo añadir una decisión
Registra el contexto, las opciones consideradas, la decisión, sus contras, la fecha y la persona que la aprueba.
