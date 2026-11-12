# Módulo RAG — políticas internas (PDF)

Responsable: Claude Code. Contrato de integración: `rag-v1` (`app/rag/interface.py`).

## Flujo

```text
PDFs ──PyMuPDF──> bloques (archivo, página, sección) ──> fragmentos (≤160 palabras, solape 40,
nunca cruzan página/sección) ──> índice JSON (BM25 + embeddings Ollama opcionales)

pregunta ──> BM25 (+ coseno de embeddings) ──RRF──> top-k ──> filtro de cobertura (abstención)
         ──> detección de conflictos ──> respuesta extractiva citada + evidencias + métricas
```

| Archivo | Función |
| --- | --- |
| `extraction.py` | PyMuPDF, detección de secciones (tamaño/negrita/numeración), páginas vacías, PDFs corruptos o cifrados, fechas de vigencia y frases "reemplaza a…" / "en caso de conflicto…" tomadas del texto |
| `chunking.py` | Fragmentación con ventana deslizante por página y sección |
| `bm25.py`, `text.py` | BM25 sin dependencias; normalización española (acentos, plurales, stopwords) |
| `embeddings.py` | Adaptador Ollama `POST /api/embed` (stdlib), registra llamadas, tokens y latencia |
| `index.py` | Construcción, persistencia (`index.json` + `.gitignore` automático), búsqueda BM25/híbrida RRF (k=60) |
| `conflicts.py` | Contradicciones numéricas (días, %, horas, $…) y de modalidad (permitido/prohibido) |
| `service.py` | `retrieve_policy`, `answer_policy_question`, abstención, respuesta extractiva |
| `interface.py` | **Interfaz estable para el backend**: `search_policies`, `rag_health` |
| `prompts.py` | Prompt fundamentado opcional para el adaptador LLM compartido |
| `evaluation.py` | hit@k, MRR, abstención, conflictos, latencia, tokens |
| `synthetic.py` | 5 PDFs **ficticios** para pruebas/demo |

## Modos de recuperación
- **BM25** si Ollama no responde (al construir o al consultar), si `HEB_RAG_EMBEDDINGS=off`, o si el modelo de consulta no coincide con el del índice. Se informa en `metrics.retrieval_mode` y `metrics.warnings`.
- **Híbrido (RRF)** cuando el índice tiene embeddings y Ollama responde con el mismo modelo.
- `HEB_RAG_EMBEDDINGS=required` hace fallar la construcción si Ollama no está disponible.

`score`: puntuación BM25 en modo bm25, puntuación RRF en modo híbrido. Solo es comparable dentro de una misma respuesta.

## Conflictos
Se muestran **ambas** evidencias. Solo se marca `resolved_by_source` cuando un documento dice explícitamente que reemplaza al otro o establece una regla de precedencia; la frase se cita en `basis`. Las fechas de vigencia se reportan pero **nunca** deciden por sí solas.

## Abstención
`insufficient_data` si ningún fragmento cubre ≥ `HEB_RAG_MIN_COVERAGE` (0.45) del peso IDF de los términos de la pregunta (o, en modo híbrido, coseno ≥ `HEB_RAG_MIN_EMBED_SIM`, 0.6). Los umbrales se ajustaron con el corpus sintético y **deben recalibrarse con los PDFs reales**.

## Configuración (variables de entorno)
`HEB_RAG_PDF_DIR` (data/policies), `HEB_RAG_INDEX_DIR` (data/rag_index), `HEB_RAG_EMBEDDINGS` (auto|off|required), `OLLAMA_BASE_URL` (http://localhost:11434), `HEB_RAG_EMBED_MODEL` (bge-m3), `HEB_RAG_TOP_K`, `HEB_RAG_MIN_COVERAGE`, `HEB_RAG_MIN_EMBED_SIM`, `HEB_RAG_CHUNK_MAX_WORDS`, `HEB_RAG_CHUNK_OVERLAP_WORDS`, `HEB_RAG_OLLAMA_TIMEOUT_S`.

## Uso

```bash
pip install -r app/rag/requirements.txt
python -m app.rag synthetic --out data/synthetic_policies
python -m app.rag build --pdf-dir data/synthetic_policies --index-dir data/rag_index
python -m app.rag query "¿Cuántos días tengo para devolver un producto?" --index-dir data/rag_index
python -m app.rag eval --cases tests/rag/fixtures/synthetic_eval.json --index-dir data/rag_index
python -m pytest tests/rag -q
HEB_RAG_OLLAMA_TESTS=1 python -m pytest tests/rag -q   # con Ollama local y `ollama pull bge-m3`
```

Con PDFs reales: cópialos a una carpeta local **no versionada** y apunta `--pdf-dir` a ella. El índice contiene pasajes de los PDFs; su carpeta incluye un `.gitignore` con `*` para no subirlo por error.

## Limitaciones
- Sin OCR: las páginas escaneadas se reportan en `warnings`/`empty_pages` y no se indexan.
- Tablas y diseños multicolumna complejos se extraen como texto plano.
- Detección de secciones heurística (tamaño de letra, negrita, numeración).
- Detección de conflictos heurística: puede omitir contradicciones expresadas sin números ni modalidad explícita, o redactadas de forma muy distinta.
- Stemming ligero; sinónimos solo se capturan en modo híbrido.
- BM25 en Python puro y coseno sin numpy: adecuado para cientos/miles de fragmentos, no para corpus grandes.
- La respuesta es extractiva (sin LLM). La redacción con LLM se hará en el orquestador compartido (`app/llm/`) usando `prompts.build_grounded_prompt`.
