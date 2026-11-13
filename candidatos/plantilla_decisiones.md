# Documento de decisiones

Nombre: [PENDIENTE: tu nombre]
Fecha: [PENDIENTE: fecha de envío]
Repositorio: https://github.com/h8hyf2rw9y-afk/HEB
Proveedor y modelos usados: Ollama local, sin APIs de pago. Llama 3.2 (3B) elige las consultas y redacta las respuestas; también probé `qwen3:4b-instruct`. `bge-m3` es opcional para embeddings. Las cifras salen de DuckDB con SQL fijo y las políticas de una búsqueda BM25 sobre los 4 PDF.

Llena cada sección en 3 a 5 líneas. No hace falta que sea prosa pulida; nos interesa el razonamiento.

## 1. Qué construí y por qué en ese orden
Primero los contratos entre piezas (FastAPI y la interfaz del RAG), para que datos, políticas e interfaz avanzaran en paralelo. Después las políticas, porque es donde un asistente inventa con más facilidad: citas con archivo, página y sección, detección de la contradicción FAQ (15 días) contra el procedimiento (30 días) y abstención cuando no hay evidencia. Luego los datos: DuckDB con consultas fijas y parametrizadas sobre ventas, inventario y tickets. Después Streamlit y la verificación en Windows y Linux. Al final el LLM local, que primero solo redactaba y ahora decide qué herramienta consultar: lo dejé para el final porque necesitaba fuentes verificadas contra las cuales validarlo.

## 2. Qué decidí no hacer
- No dejar que el modelo escriba SQL ni lea los archivos: solo puede llamar a dos herramientas verificadas, una para los datos de tienda y otra para buscar en las políticas.
- No mostrar texto del modelo que cambie cifras, tiendas, SKUs, fechas o citas: en ese caso se muestra la salida verificada de la herramienta.
- No resolver contradicciones por fecha ni por "el más reciente": solo cuando el propio documento dice cuál prevalece.
- No hacer pronósticos ni responder fuera del periodo de los datos (junio–agosto de 2026; inventario al 1 de septiembre).
- No desplegar en la nube, no usar APIs de pago y no subir datos, índices ni resultados locales al repositorio.

## 3. Cómo sé que funciona
- Políticas: 49 preguntas independientes, escritas antes de ajustar nada: 46 correctas, 0 abstenciones falsas y citas verificadas contra el PDF. En el conjunto de desarrollo da 63 de 63, pero esa cifra no mide generalización.
- Datos: las cifras de la demo se recalculan en las pruebas con Python puro, sin DuckDB, y coinciden. Por ejemplo, Tienda Valle Norte vendió $2,163,062.36 MXN en agosto de 2026.
- LLM: las pruebas simulan Ollama y comprueban que una cifra alterada, una tienda cambiada, una cita inventada o el plazo de 30 días convertido en 15 se rechazan y caen a la respuesta verificada.
- GitHub Actions corre todas las pruebas en Windows y Ubuntu. [ACTUALIZAR: resultado final de CI al cerrar el PR]
- Con el modelo real hice una prueba manual en Windows, pero falta la evaluación completa con Llama 3.2. [PENDIENTE: resultados de `eval_llm_modes.py` y `eval_agent_conversations.py`]

## 4. Costo por interacción y cómo lo medí
- Costo monetario por token: $0, porque el modelo corre en Ollama local.
- Costo real: memoria, CPU o GPU y tiempo. En mi prueba manual con `qwen3:4b-instruct`, una consulta del agente tardó de 17 a 147 segundos. Sin modelo, el modo determinista responde en unos 3 ms.
- Medición: cada respuesta de `/chat` trae llamadas al LLM, tokens de entrada y salida (los contadores reales de Ollama), latencia de generación y latencia total. Los embeddings se cuentan aparte.
- Una pregunta del agente hace al menos dos llamadas: una para elegir la herramienta y otra para redactar. [PENDIENTE: tokens y latencia medianos de Llama 3.2 desde `data/eval_llm_modes.json`]

## 5. Qué haría con dos semanas más
- Medir Llama 3.2, Llama 3.1 y Qwen con las mismas preguntas, y elegir el modelo por tasa de respuestas validadas, latencia y tokens, no por impresión.
- Bajar la latencia: menos rondas del agente, prompts más cortos y respuestas por streaming.
- Mejorar la redacción sin relajar la validación: hoy muchas redacciones se descartan por omitir filas, y eso degrada a texto determinista.
- Comparar embeddings `bge-m3` contra BM25 en las 49 preguntas independientes, porque los 3 fallos actuales son de vocabulario.
- Validar con el área responsable las dos inconsistencias entre documentos que el sistema no detecta.
