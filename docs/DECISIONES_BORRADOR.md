# Borrador para `candidatos/plantilla_decisiones.md`

> **Borrador preparado por Claude Code a partir del repositorio (2026-11-12).** No sustituye a la plantilla original: la persona que entrega lo revisa, lo reescribe con su voz y lo copia a `candidatos/plantilla_decisiones.md`. Los `[PENDIENTE …]` solo se rellenan con datos reales. La plantilla pide 3 a 5 líneas por sección.

Nombre: `[PENDIENTE: persona que entrega]`
Fecha: `[PENDIENTE]`
Repositorio: `[PENDIENTE: liga a GitHub o zip]`
Proveedor y modelos usados: `[PENDIENTE: ADR-010 en docs/DECISIONS.md]`. A la fecha, el sistema **no usa ningún LLM**. Las políticas se responden con recuperación BM25 local y respuestas extractivas citadas. Los embeddings con Ollama están implementados, pero no se han probado con un modelo real.

## 1. Qué construí y por qué en ese orden
Primero la base y los contratos entre piezas (FastAPI, contrato `rag-v1`), para que interfaz, datos y políticas avanzaran en paralelo sin pisarse. Después, el buscador de políticas: era lo único que podía inventar texto si se hacía mal, y la gerente pidió citas y "que no me invente". Lo evalué con los 4 PDF hasta citar documento, página y sección verificables, detectar la contradicción real FAQ (15 días) frente al procedimiento (30 días) y abstenerse con motivo. En paralelo, la interfaz Streamlit y la integración con `/chat`. `[PENDIENTE Codex: consultas DuckDB sobre ventas, inventario y tickets, y en qué orden]`.

## 2. Qué decidí no hacer
- No usar un LLM para redactar las respuestas de políticas: preferí respuestas extractivas verificables a respuestas fluidas no verificables.
- No resolver contradicciones por fecha ni por "el documento más reciente": solo cuando la propia fuente declara cuál prevalece.
- No subir datos ni índices al repositorio; tampoco carga de archivos desde la interfaz.
- No desplegar en la nube: todo corre en local.
- `[PENDIENTE: decisiones de Codex sobre datos, por ejemplo pronósticos y SQL libre]`

## 3. Cómo sé que funciona
- **Políticas, 49 preguntas independientes** (escritas antes de los últimos cambios): 46 correctas, 93 % de abstenciones correctas, **0 abstenciones falsas** y 100 % de citas verificadas palabra por palabra contra la página del PDF.
- **Conjunto de desarrollo:** 63/63.
- **Pruebas automatizadas:** 188 pasan (RAG, backend y documentación), y `scripts/verify_delivery.py` comprueba la instalación de punta a punta.
- **Fallos conocidos y documentados:** 3 preguntas independientes y 2 inconsistencias entre documentos no detectadas (`docs/LIMITATIONS.md`).
- `[PENDIENTE Codex: set de preguntas analíticas con respuestas calculadas de forma independiente y sus resultados]`

## 4. Costo por interacción y cómo lo medí
Hoy, una pregunta de política cuesta **0 tokens**: no hay llamadas a un LLM ni embeddings, y la latencia de `/chat` es de ~2–6 ms (medido con `scripts/verify_delivery.py`). `[PENDIENTE: modelo elegido; mediana y p90 de tokens de entrada y salida por interacción típica, medidos con las métricas de /chat; precio del proveedor en la fecha de la medición, o "local, $0" con el hardware usado]`. Método en `docs/TOKEN_COSTS.md`.

## 5. Qué haría con dos semanas más
- **Medir embeddings reales** (Ollama `bge-m3`) frente a BM25 en el conjunto independiente. Los 3 fallos actuales son de vocabulario; podrían mejorar, pero hay que medirlo.
- **Escribir un conjunto independiente nuevo,** más grande, con preguntas reales de gerentes, para calibrar el umbral de abstención (0.45 = 0.50 con los datos actuales).
- **Detectar inconsistencias que requieren razonar sobre fechas** (rebaja a 1 día frente a retiro un día antes) y validarlas con el área responsable.
- **Incorporar un LLM local solo para redactar sobre la evidencia citada,** midiendo tokens por interacción.
- `[PENDIENTE Codex: siguientes pasos de datos]`
