# Documento de decisiones

Nombre: Emiliano Gonzalez Romo
Fecha: 13 de noviembre de 2026
Repositorio: https://github.com/h8hyf2rw9y-afk/HEB
Proveedor y modelos usados: Ollama local, sin APIs de pago. `qwen3:4b-instruct` redacta las respuestas y, en preguntas ambiguas, elige la consulta; antes medí `llama3.2`. Las cifras salen de DuckDB con SQL fijo y parametrizado; las políticas, de una búsqueda BM25 sobre los 4 PDF (embeddings `bge-m3` opcionales).

Llena cada sección en 3 a 5 líneas. No hace falta que sea prosa pulida; nos interesa el razonamiento.

## 1. Qué construí y por qué en ese orden
- Primero los contratos entre FastAPI, datos y políticas, para avanzar las piezas en paralelo.
- Después las políticas, donde un asistente inventa con más facilidad: citas con archivo, página y sección, la contradicción FAQ (15 días) contra el procedimiento (30 días) y abstención sin evidencia.
- Luego los datos con DuckDB y consultas fijas, Streamlit y pruebas en Windows y Linux.
- Al final el LLM local, porque necesitaba fuentes verificadas contra las cuales validarlo. En las pruebas reales vi que pedirle al modelo que eligiera la herramienta costaba una llamada lenta, así que ahora el sistema consulta primero la evidencia y el modelo solo redacta.

## 2. Qué decidí no hacer
- No dejar que el modelo escriba SQL, lea archivos ni decida qué política prevalece: solo redacta sobre resultados verificados.
- No mostrar texto del modelo que cambie cifras, tiendas, fechas o citas, ni que afirme causas que los datos no registran: en ese caso se muestra la respuesta verificada.
- No explicar «por qué vendió más» con promociones, ubicación o clientes: muestro las diferencias que sí están en los CSV y digo que no prueban causas.
- No hacer pronósticos ni responder fuera de los datos (ventas de junio a agosto de 2026; inventario al 1 de septiembre).
- No desplegar en la nube ni usar APIs de pago; no subir datos, índices ni resultados locales al repositorio.

## 3. Cómo sé que funciona
- Políticas: 46 de 49 preguntas independientes, escritas antes de ajustar nada, con 0 abstenciones falsas y citas verificadas contra el PDF.
- Datos: las cifras se recalculan en las pruebas con Python puro, sin DuckDB, y coinciden (por ejemplo, Tienda Valle Norte: $2,163,062.36 MXN en agosto de 2026).
- 433 pruebas automáticas pasan en Windows y Ubuntu, incluida una conversación con las siete preguntas que fallaron en mi prueba real; con Ollama simulado se comprueba que una cifra alterada, una tienda cambiada o una cita inventada nunca llegan al usuario.
- Con Qwen real en mi laptop, las siete preguntas como una sola conversación: 7 de 7 con las cifras y fuentes correctas, una llamada al modelo por pregunta, y 6 de 7 redactadas por Qwen; en la de la licuadora su redacción no pasó la validación y se mostró el texto verificado.

## 4. Costo por interacción y cómo lo medí
- Costo real por token: $0, porque el modelo corre local. El costo real es tiempo de cómputo: con Qwen en mi laptop la mediana fue de 30 s por pregunta (de 9 a 91 s, a unos 6 a 9 tokens/s), contra milisegundos del modo determinista.
- Cada respuesta de `/chat` reporta llamadas al LLM, tokens de entrada y salida (contadores reales de Ollama) y el tiempo por etapa: carga, lectura del prompt y generación.
- Para bajar el costo, una pregunta clara hace una sola llamada en lugar de dos y los resúmenes de datos tienen tope de 220 tokens: en las siete preguntas pasé de 13 a 7 llamadas; el ranking bajó de 57 a 35 s y el seguimiento de julio de 55 a 9 s.
- Simulé el costo con la API de Anthropic usando los tokens medidos con Qwen: mediana de 724 tokens de entrada y 45 de salida, unos $0.0001 por pregunta con Haiku 5.5, $0.0019 con Sonnet 5.5 y $0.0038 con Opus 5.5 ($0.12, $2.40 y $4.80 por 1,000 preguntas). Es una aproximación: Claude cuenta tokens distinto y cobra su razonamiento como salida.

## 5. Qué haría con dos semanas más
- Medir Qwen, Llama 3.2 y Llama 3.1 con las mismas preguntas y elegir por respuestas validadas, latencia y tokens.
- Reemplazar el enrutamiento por reglas con un plan de consulta tipado y validado, para que frases nuevas no tomen otra ruta.
- Bajar la latencia de las respuestas largas de políticas, que exigen reescribir todas las citas: respuestas por streaming y prompts más cortos, sin relajar la validación.
- Comparar embeddings `bge-m3` contra BM25 en las 49 preguntas independientes y validar con el área responsable las dos inconsistencias entre documentos que el sistema no detecta.
