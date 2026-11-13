# Capa generativa local (Ollama + Llama 3.2)

Dos modos, elegidos con `HEB_LLM_MODE` cuando `HEB_LLM_ENABLED=true`:

**`agente` (por defecto).** Llama 3.2 decide qué consulta hacer. Recibe la pregunta y dos herramientas, `consultar_datos_tienda` (DuckDB con SQL fijo) y `buscar_politicas` (RAG de los PDF), elige cuál llamar y con qué texto, y redacta la respuesta con lo que devolvieron. El modelo nunca escribe SQL ni lee los archivos: solo puede invocar esas funciones verificadas.

```text
pregunta ─► Llama 3.2 ─► tool_call ─► DuckDB / RAG (código verificado) ─► Llama 3.2 redacta ─► validación ─► respuesta
                │ no llamó herramienta / Ollama caído          │ abstención o validación fallida
                ▼                                              ▼
       ruta determinista                          salida verificada de la herramienta, tal cual
```

Protecciones del agente: si el modelo cambia el mes, año, tienda o SKU al pasar la consulta a la herramienta, se ejecuta la pregunta original; como máximo 3 consultas por pregunta; una abstención de la herramienta se muestra tal cual; una salida de herramienta con instrucciones sospechosas no se devuelve al modelo; `metrics.tool_calls` muestra qué consultó el modelo y con qué texto.

**`redaccion`.** El enrutador determinista consulta y el modelo solo reescribe la respuesta verificada.

En ambos modos la redacción pasa por el mismo validador y cualquier fallo devuelve texto verificado, nunca texto inventado. Saludos, diagnóstico de Ollama y conversación general no pasan por el agente.

## Archivos
| Archivo | Función |
| --- | --- |
| `config.py` | `LLMSettings.from_env()`: `HEB_LLM_ENABLED`, `OLLAMA_BASE_URL`, `HEB_LLM_MODEL`, `HEB_LLM_TIMEOUT_S`, `HEB_LLM_MAX_OUTPUT_TOKENS`, `HEB_LLM_MODE` (`agente` o `redaccion`) |
| `ollama_client.py` | `POST {OLLAMA_BASE_URL}/api/chat` sin streaming, `temperature=0`, `seed=42`. Errores tipados: no disponible, timeout, respuesta vacía, JSON o HTTP inválidos. Lee los contadores reales `prompt_eval_count` y `eval_count`. |
| `prompts.py` | Mensaje de sistema con las reglas y mensaje de usuario con la respuesta verificada delimitada en `<respuesta_verificada>` como **datos**, más `<contradicciones>` cuando el RAG las detectó |
| `validation.py` | Comprobaciones conservadoras sobre la redacción (ver abajo) |
| `agent.py` | `ToolAgent.run(...)`: bucle de llamadas con herramientas (`tools` de `/api/chat`), protecciones y fallback; devuelve `AgentOutcome` |
| `service.py` | `GroundedWriter.rewrite(...)`: orquesta, nunca lanza excepciones, devuelve `GenerationOutcome` con la respuesta final, métricas y motivo de fallback |

## Activar y desactivar
Por defecto está **desactivada** (`HEB_LLM_ENABLED` ausente o `false`): `/chat` se comporta exactamente como antes. Se configura al arrancar la API.

PowerShell, activar:
```powershell
$env:HEB_LLM_ENABLED = "true"
$env:HEB_LLM_MODEL = "llama3.2"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:HEB_LLM_TIMEOUT_S = "60"
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --port 8000
```
Desactivar: `$env:HEB_LLM_ENABLED = "false"` (o `Remove-Item Env:HEB_LLM_ENABLED`) y reiniciar la API. `GET /health/llm` muestra la configuración activa sin llamar a Ollama.

## Qué se valida antes de mostrar una redacción
La redacción se descarta (y se devuelve la determinista, con el motivo en `metrics.warnings`) si:
- falta o cambia cualquier número, monto, porcentaje, fecha, hora, mes o SKU de la respuesta verificada ("48 horas" ≡ "cuarenta y ocho horas" se acepta; "30" → "15" o "35" no);
- aparece un número, fecha o SKU que no está en la respuesta verificada ni en la pregunta;
- en respuestas de datos, una cifra queda junto a otra entidad ("Entidad: cifras" intercambiadas);
- falta una cita `[archivo.pdf, p. N, sección: …]` o aparece una que no existe, o se menciona un archivo `.pdf`/`.csv` o código de documento (`PRO-SAC-021`) ajeno a la evidencia;
- en políticas, las cifras se separan de la cita que las respalda, o un párrafo largo queda sin cita;
- se omite la advertencia de contradicción; en una contradicción resuelta por la fuente no se dice "prevalece <documento>" o se nombra otro; en una sin resolver se elige un ganador;
- se pierden negaciones ("no", "nunca", "prohibido") o restricciones ("solo", "únicamente", "exclusivamente");
- declara falta de información cuando la respuesta verificada sí la tenía;
- contiene SQL o código, o reproduce instrucciones inyectadas ("ignora las instrucciones…");
- es vacía o más del triple de larga que la original.

Las respuestas con estado `insufficient_data`, `needs_clarification` o `error` **nunca** se envían al modelo. Si la propia evidencia contiene texto con forma de instrucción, tampoco se envía.

**Lo que la validación no garantiza:** que no haya alucinaciones. Una redacción puede añadir una afirmación cualitativa sin números ni citas, o cambiar el sentido sin tocar ningún elemento protegido. Las citas y las cifras siguen siendo la referencia que el usuario debe poder comprobar.

## Métricas por consulta (`/chat` → `metrics`)
| Campo | Qué mide |
| --- | --- |
| `llm_calls`, `llm_model` | Llamadas al modelo generativo y modelo usado (0 / `null` si no se llamó) |
| `input_tokens`, `output_tokens`, `total_tokens` | Contadores reales de Ollama (`prompt_eval_count`, `eval_count`). También se contabilizan cuando la redacción se descarta: la llamada ocurrió. |
| `generation_latency_ms` | Tiempo de la llamada a `/api/chat` |
| `latency_ms` | Toda la petición: SQL o recuperación + generación + validación |
| `generated` | `true` solo si la respuesta mostrada la escribió el modelo |
| `fallback_reason` | `llm_desactivado`, `estado_<x>_no_se_redacta`, `ollama_no_disponible`, `timeout`, `respuesta_vacia`, `respuesta_invalida`, `validacion_fallida`, `evidencia_con_instrucciones_sospechosas` o `null` |
| `embedding_calls`, `embedding_input_tokens`, `embedding_latency_ms` | Embeddings de la recuperación híbrida, contados aparte |
| `tool_calls` | Modo agente: herramienta elegida, consulta ejecutada, estado, `duration_ms`; `retried_with_original_question` si la reformulación del modelo no encontró nada y se reintentó con la pregunta original |
| `timings` | Tiempo por etapa: carga del modelo, lectura del prompt, generación, tokens/s, herramientas y resto (ruteo, SQL, recuperación, validación) |

**Costo:** la inferencia es local, así que no hay costo por token de API. Sí hay costo de ejecución local: el modelo ocupa memoria (`load_duration` de Ollama) y CPU o GPU durante segundos por respuesta, frente a milisegundos del modo determinista. Los tokens de embeddings son otro consumo distinto, que se reporta por separado.

## Resultados reales y modo rápido de datos (noviembre 2026)

Una evaluación manual en Windows con `qwen3:4b-instruct` confirmó que el LLM puede tardar **17 a 147 segundos** por consulta y que muchas redacciones fueron descartadas por conservar solo parte de los resultados o alterar asignaciones de cifras a tiendas. El fallback verificado fue correcto: **no** debemos relajar la validación de montos para que el texto parezca más natural. Al preguntar por enero, la falta de datos es correcta: `ventas.csv` solo cubre junio–agosto de 2026. Los seguimientos con periodos ausentes ya no necesitan gastar llamadas al LLM para esa respuesta.

El flujo permite activar una vía rápida opcional **solo para rankings exactos de tiendas y comparaciones de productos entre dos tiendas**. DuckDB ejecuta la plantilla SQL aprobada y devuelve cifras y fuentes directamente; las demás consultas continúan usando Ollama y sus comprobaciones. Actívala **antes de iniciar FastAPI**:

```powershell
$env:HEB_FAST_ANALYTICS = "true"
$env:HEB_LLM_ENABLED = "true"
$env:HEB_LLM_MODE = "agente"
$env:HEB_LLM_MODEL = "qwen3:4b-instruct"
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --port 8000
```

Con la vía rápida, `metrics.llm_calls=0` para esas consultas es **esperado**; no indica que Ollama esté caído. Para medir el desempeño completo de selección de herramientas del agente sin este atajo, usa `HEB_FAST_ANALYTICS=false` y reinicia FastAPI. Un conteo de productos por tienda se responde siempre del inventario usando SKU únicos por tienda; no es una medición de punto de reorden ni demuestra disponibilidad de unidades. Las preguntas causales como «¿por qué vendió más?» se pueden desglosar por diferencias observadas de productos, pero estas correlaciones no prueban causas reales.

Una pregunta que confunde `cambio de producto` con `reembolso` debe recuperar evidencia de ambos conceptos, o pedir precisión sin afirmar una equivalencia inexistente. También se reconoce la grafía «rembolso».

## Redacción obligatoria del modelo cuando está disponible

Con `HEB_LLM_ENABLED=true`, las consultas conocidas del catálogo y el modo
`HEB_FAST_ANALYTICS=true` **no omiten** la generación. DuckDB/RAG obtienen
primero la evidencia verificada y Ollama redacta la explicación: normalmente
una sola llamada al modelo, sin otra llamada adicional para elegir herramienta.
El agente conserva el flujo de selección mediante herramientas para preguntas
que requieren planificación.

En resúmenes de ventas por tienda, si el texto del modelo omite tiendas del
ranking, el sistema puede conservar la introducción redactada y adjuntar
`**Detalle verificado de los CSV:**` con todas las cifras inalteradas; vuelve
a validar el conjunto y rechaza cifras nuevas, valores incorrectos o fuentes
inventadas. **No aplica** esta flexibilidad a SKU individuales, inventario
por unidad ni políticas: esas respuestas conservan validación estricta.

Si Ollama falla, omite datos críticos, cambia reglas o la validación detecta
un problema, se entrega la respuesta verificable como *fallback* y
`metrics.generated=false` junto con `fallback_reason`; no se afirma que
Ollama escribió una respuesta que no escribió. Tampoco se redactan
predicciones o políticas sin fuentes verificadas. Esto prioriza seguridad
ante la garantía absoluta de generación.

## Latencia y tiempos de espera en Streamlit

El agente puede invocar Ollama varias veces de forma secuencial (selección de herramienta + redacción, y hasta cuatro rondas). El frontend ya no impone un límite fijo de 30 s: utiliza `HEB_CHAT_TIMEOUT_S` (por defecto 540 s) para la petición completa. Puedes ajustarlo antes de iniciar Streamlit:

```powershell
$env:HEB_CHAT_TIMEOUT_S = "540"
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Si aún aparece `ReadTimeout`, revisa en la terminal FastAPI cuánto tardó la petición, el estado de Ollama y los tokens. No confundas un timeout de la interfaz con una caída confirmada de FastAPI. Las rutas de conteo de tiendas y listado de fuentes se contestan desde el catálogo local (sin esperar al modelo); preguntas generales sobre la identidad del asistente no consultan DuckDB.

## Evaluar conversaciones de varios turnos

Con FastAPI iniciado en modo agente y Ollama operativo, ejecuta en otra terminal:

```powershell
.\.venv\Scripts\python.exe scripts\eval_agent_conversations.py --out data\eval_agent_conversations.json
```

El script prueba seguimientos sobre devoluciones, tarjeta original y ventas, con `conversation_id` aislado por episodio. Registra selección de herramientas, respuesta, evidencia, estados, fallbacks, tokens y percentiles de latencia. **La exactitud de citas y políticas requiere revisión humana:** estos indicadores no garantizan ausencia de alucinaciones.

## Comparar los modos en tu equipo
El script ejecuta sobre las mismas 12 preguntas el Modo A (determinista), el Modo B (redacción) y el Modo C (agente); activa el LLM por sí mismo en B y C.
```powershell
$env:HEB_LLM_MODEL = "llama3.2"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
.\.venv\Scripts\python.exe scripts\eval_llm_modes.py --out data\eval_llm_modes.json
```
Prueba opcional contra Ollama real (en CI siempre se omite):
```powershell
$env:HEB_LLM_REAL_TESTS = "1"
.\.venv\Scripts\python.exe -m pytest tests\llm\test_real_ollama.py -q -s
Remove-Item Env:HEB_LLM_REAL_TESTS
```

## Rutas conversacionales y límites de evidencia

El endpoint `/chat` distingue tres formas de respuesta:

- **Consulta H-E-B fundamentada:** datos de `tiendas.csv`, otros CSV o RAG de políticas. Las cifras se calculan de manera determinista, y cualquier reescritura generativa debe pasar la validación. `metrics.answer_mode` indica `grounded_data` o `grounded_policy`.
- **Conversación general (opcional):** saludos, identidad y preguntas generales pueden pasar por `app/llm/general.py` cuando `HEB_LLM_ENABLED=true`. **No están respaldadas por los CSV ni los PDF**, y el resultado puede contener errores factuales. Se marca `answer_mode=general_unverified` cuando interviene esa ruta.
- **Consultas sobre capacidades, fuentes, salud de Ollama y casos sin contexto:** se atienden de modo determinista o solicitan aclaración. Los temas laborales/operativos no deben acceder al chat generativo libre; ausencia prolongada, incapacidad y posibles medidas disciplinarias requieren validación del área de Recursos Humanos y políticas aplicables, no inferencias del modelo.

**Memoria conversacional local y acotada:** `conversation_id` vincula las últimas **6 preguntas** y la **última respuesta fundamentada** de esa sesión durante **30 minutos de inactividad**, con fuentes CSV/PDF y un máximo de **1,800 caracteres de respuesta** y 12 referencias. Solo se conserva una respuesta cuando la operación produjo `status=answered` con fuentes verificadas; las afirmaciones del chat general, las abstenciones y los errores no actualizan esa memoria factual. El estado es volátil, de un único proceso FastAPI, con un máximo de 128 sesiones; desaparece tras un reinicio y `POST /chat/reset` lo elimina.

**Preguntas de seguimiento:** el resolver de `app/api/followup.py` usa exclusivamente resultados respaldados por CSV/PDF para completar referencias acotadas y crear una **nueva consulta**, no para repetir valores antiguos. Ejemplos: «¿Qué tienda vendió más en agosto?», «¿Y cómo le fue en julio?» recalcula las ventas del líder contra julio; «Entonces compara sus productos con los de Cumbres» ejecuta SQL parametrizado para mostrar los cinco productos de mayor venta neta por cada una de las dos tiendas en el periodo conservado. «¿Y en julio?» conserva el ranking de todas las tiendas y lo ejecuta para ese mes. El soporte de comparaciones entre tiendas requiere que se puedan identificar las dos tiendas del catálogo; no se interpretan causalidades a partir de ventas solamente.

Para preguntas de seguimiento de políticas como «¿Y si no tiene el ticket?», el sistema puede ampliar la nueva búsqueda usando el tema de una pregunta anterior que sí recuperó evidencia PDF. **Siempre vuelve a consultar el índice RAG**; no considera la respuesta guardada como sustituto de la política y no traspasa evidencia entre conversaciones. El ToolAgent puede recibir ese resultado previo como contexto breve para interpretar pronombres, pero las cifras y reglas nuevas deben pasar nuevamente por DuckDB/RAG y sus comprobaciones. El campo `metrics.resolved_question` permite auditar cuándo se completó una referencia.

**Límites funcionales:** la resolución de referencias es deliberadamente conservadora; preguntas ambiguas que no admiten un único producto, tienda o periodo requieren aclaración. El motor DuckDB usa plantillas de SQL verificadas: comprender un enunciado no implica que cualquier análisis estadístico exista ya como herramienta. Los PDF FAQ están indexados como fragmentos para búsqueda; no existe una base separada de respuestas previamente aprobadas. El sistema puede recuperar su contenido y citarlo, pero no garantiza que el modelo local entienda correctamente todas las formulaciones.

**Límites:** en esta demo local no hay autenticación, sincronización entre réplicas ni memoria permanente; no debe utilizarse como almacén de datos personales o sistema multiusuario productivo. En una implementación real usar un almacén de sesiones autenticado y controles de retención/privacidad.

**Fidelidad del agente:** `app/llm/agent.py` valida claves de argumentos de las herramientas, protege SKU, periodos y conceptos críticos como «ticket de compra» frente a «ticket de soporte», y no presenta la mitad verificable de una solicitud mixta como si todo estuviera resuelto. El resultado parcial incluye la abstención y queda como `needs_clarification`. El modelo no obtiene permiso para inventar datos, escribir SQL ni inferir políticas ausentes.

**Transparencia:** Streamlit muestra un aviso visible cuando la respuesta es conversación general no verificada. `metrics.generated=true` por sí solo **no significa** que la respuesta haya sido contrastada contra datos empresariales. No se deben presentar las respuestas generales como políticas, procedimientos o criterios de Recursos Humanos de H-E-B.
