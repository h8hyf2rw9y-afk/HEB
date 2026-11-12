# Costo por interacción

Requisito del brief: *"Que reportes cuánto cuesta en tokens una interacción típica, con el modelo que hayas elegido."* Contexto del mismo brief: *"Tú eliges proveedor y modelo […] No evaluamos cuál elegiste, sino cómo lo usas y si sabes lo que cuesta."*

## Decisión: sin LLM generativo
La versión entregada **no usa ningún modelo generativo**. Es una decisión de diseño intencional (ADR-010 en [`DECISIONS.md`](DECISIONS.md)):

| Tipo de pregunta | Cómo se responde | Modelo |
| --- | --- | --- |
| Datos (ventas, inventario, tickets) | SQL fijo y parametrizado con DuckDB sobre los CSV (`app/tools/analytics.py`) | Ninguno |
| Políticas (4 PDF) | Recuperación BM25 y respuesta **extractiva**: oraciones copiadas del PDF con su cita (`app/rag/`) | Ninguno |
| Enrutamiento entre datos y políticas | Reglas léxicas (`looks_like_data`) | Ninguno |

Motivos:
1. **Verificabilidad.** Cada cifra sale de una consulta SQL reproducible y cada oración de política existe en el PDF citado (verificado palabra por palabra en 112 preguntas). No hay texto generado que pueda inventar una cifra, que es lo que la gerente pidió evitar.
2. **Costo.** Sin tokens que pagar ni hardware para un modelo local.
3. **Reproducibilidad.** Los mismos datos dan la misma respuesta, sin variación por temperatura ni por versión de modelo.

Lo que se pierde a cambio: las respuestas no se redactan con naturalidad, el catálogo de preguntas de datos es acotado y el enrutamiento por reglas comete errores (ver [`LIMITATIONS.md`](LIMITATIONS.md)).

## Costo medido
Medido con `/chat` en un entorno limpio (Linux, Python 3.13; 5 repeticiones, mediana de latencia):

| Pregunta | Estado | Llamadas LLM | Tokens LLM | Llamadas de embeddings | Latencia `/chat` |
| --- | --- | --- | --- | --- | --- |
| ¿Cuántos días tiene un cliente para devolver una licuadora? | answered | 0 | 0 | 0 | 4.2 ms |
| ¿Cómo se hace la devolución si el cliente pagó con tarjeta y no trae la tarjeta original? | answered | 0 | 0 | 0 | 4.5 ms |
| ¿Cuál es la tolerancia de merma en panadería? | answered | 0 | 0 | 0 | 2.9 ms |
| ¿Cuál es la tolerancia de merma para electrónica? | insufficient_data | 0 | 0 | 0 | 2.8 ms |
| ¿Qué tienda vendió más en agosto de 2026? | answered | 0 | 0 | 0 | 62 ms |
| ¿Qué productos de la tienda T01 están por debajo del punto de reorden? | answered | 0 | 0 | 0 | 45 ms |
| ¿Cuántos tickets de prioridad Crítica siguen abiertos y de qué tiendas son? | answered | 0 | 0 | 0 | 29 ms |
| ¿Cuánto vamos a vender en diciembre? | insufficient_data | 0 | 0 | 0 | 0.3 ms |

**Costo variable por interacción: 0 llamadas a LLM, 0 tokens y $0 MXN por tokens.** Los únicos costos son de cómputo local: CPU, unos milisegundos por pregunta y la construcción única del índice, de 2–8 s.

Estas cifras son **medidas**, no estimadas. No se reporta el costo de ningún modelo generativo porque no se ejecutó ninguno.

## ¿Satisface el requisito del brief?
**Interpretación adoptada:** el brief deja libre la elección de "proveedor y modelo" y evalúa "si sabes lo que cuesta". Elegir no usar un modelo generativo y medir que la interacción cuesta 0 tokens es una respuesta completa y honesta a "cuánto cuesta en tokens".

**Riesgo de interpretación:** la redacción "con el modelo que hayas elegido" da por hecho que hay un modelo, y la sección "Sobre el modelo que uses" sugiere que el ejercicio espera un uso de LLM. Un evaluador podría leer la ausencia de LLM como requisito no cubierto, o como que el asistente no entiende lenguaje natural abierto.

**Recomendación:**
1. Declarar la decisión de forma explícita en `plantilla_decisiones.md` (sección 4) y en el video, con esta tabla.
2. Aprovechar la invitación del brief ("las preguntas sobre el alcance son bienvenidas") para **confirmar con la reclutadora** si una solución sin LLM generativo es aceptable o si esperan ver el costo de un modelo.
3. Si piden un modelo, la opción de menor riesgo es usarlo solo para **redactar sobre la evidencia ya obtenida**, sin que calcule cifras ni elija el SQL: `app.rag.prompts.build_grounded_prompt(answer)` arma ese prompt. `/chat` ya expone `input_tokens`, `output_tokens`, `total_tokens` y `llm_calls`, así que el costo se mediría con el contador del proveedor (`prompt_eval_count` y `eval_count` en Ollama, `usage` en una API) y se reportarían la mediana y el p90 por interacción. **Nada de esto está implementado ni medido.**

## Embeddings (opcional, no activado)
El modo híbrido del RAG puede usar embeddings de Ollama (`bge-m3`) y registra `embedding_input_tokens`. **No se ha ejecutado con un modelo real** (ver [`LIMITATIONS.md`](LIMITATIONS.md)). Con un modelo local, el costo monetario por token sería $0; se reportarían tokens, latencia y hardware.
