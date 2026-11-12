# Costo en tokens por interacción

El brief pide: *"Que reportes cuánto cuesta en tokens una interacción típica, con el modelo que hayas elegido."*

## Estado real (2026-11-12, `main` en `5c9fcc6`)
- **No hay ningún LLM en el sistema.** No existe `app/llm/`, y `/chat` no llama a ningún modelo.
- Las respuestas de políticas son **extractivas**: se copian oraciones de los PDF con su cita. No se generan.
- Las preguntas analíticas todavía no se responden (pendiente de Codex, h8hyf2rw9y-afk/HEB#11).
- **Embeddings:** desactivados en la configuración verificada (BM25). El modo híbrido con Ollama existe, pero **no se ha ejecutado con un modelo real**.

### Medición real con lo que existe
Ejecutado con `python scripts/verify_delivery.py --json` en un entorno limpio (contenedor Linux, Python 3.13):

| Pregunta | Estado | Llamadas LLM | Tokens (entrada + salida) | Llamadas de embeddings | Latencia `/chat` |
| --- | --- | --- | --- | --- | --- |
| ¿Cuántos días tiene un cliente para devolver una licuadora? | answered | 0 | 0 | 0 | ~6 ms |
| ¿Cómo se hace la devolución si el cliente pagó con tarjeta y no trae la tarjeta original? | answered | 0 | 0 | 0 | ~3 ms |
| ¿Cuál es la tolerancia de merma en panadería? | answered | 0 | 0 | 0 | ~3 ms |
| ¿A qué hora debe llegar el gerente para la apertura de la tienda? | answered | 0 | 0 | 0 | ~3 ms |
| ¿Cuál es la tolerancia de merma para electrónica? | insufficient_data | 0 | 0 | 0 | ~2 ms |

**Costo actual por interacción de política: 0 tokens y 0 pesos, porque no se usa modelo.** Esto no responde al requisito "con el modelo que hayas elegido": todavía no hay modelo elegido.

## Decisión pendiente (persona que entrega + Codex)
Hay dos caminos defendibles; los dos hay que **medirlos**, no estimarlos.

1. **Sin LLM para políticas, con LLM solo para enrutar y redactar las analíticas.** El costo de las políticas sigue en 0 tokens; el de las analíticas sale de las llamadas reales que registre `/chat`.
2. **Con LLM que redacte sobre la evidencia.** El RAG ya ofrece `app.rag.prompts.build_grounded_prompt(answer)`, que arma un prompt con los pasajes numerados. Cada pregunta costaría el prompt (pasajes + instrucciones) más la respuesta. Se mide con el contador del proveedor (por ejemplo, `prompt_eval_count` y `eval_count` de Ollama, o `usage` de una API).

En ambos casos `/chat` ya devuelve `metrics.input_tokens`, `output_tokens`, `total_tokens` y `llm_calls`. Falta que una capa LLM los llene.

## Cómo medirlo cuando exista el modelo
1. Fijar el modelo y su versión (por ejemplo, el nombre exacto de Ollama o de la API) y anotarlo en `DECISIONES_BORRADOR.md`.
2. Ejecutar un conjunto fijo de preguntas "típicas" mixtas (políticas + analíticas; las del guion sirven) y guardar `metrics` de cada respuesta.
3. Reportar la **mediana** y el **p90** de tokens de entrada, de salida y totales por interacción, además de las llamadas por pregunta.
4. Costo en dinero: tokens × precio publicado del proveedor **en la fecha de la medición**, citando la fuente del precio. Con un modelo local (Ollama), el costo monetario por token es 0; se reportan los tokens, el hardware y la latencia.
5. Si se activan los embeddings, sumar aparte `embedding_input_tokens` (el adaptador de Ollama lo registra) para la consulta y para la indexación única.

## Lo que **no** debe hacerse
- Reportar tokens o precios de un modelo que no se haya ejecutado.
- Estimar tokens "por número de palabras" y presentarlo como medición.
- Presentar como costo del sistema la latencia de BM25, que no usa tokens.
