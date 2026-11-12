# Capa generativa local (Ollama + Llama 3.2)

Redacta en español natural las respuestas que **ya calcularon y verificaron** DuckDB o el RAG. El modelo no consulta datos, no genera SQL, no recupera documentos ni decide qué política prevalece. Si la redacción altera algo, o si Ollama no responde, el usuario recibe la respuesta determinista original.

```text
pregunta ─► FastAPI ─► DuckDB o RAG ─► respuesta verificada ─► Llama 3.2 (/api/chat) ─► validación ─► respuesta
                                             │                                            │ falla
                                             └────────────── fallback determinista ◄──────┘
```

## Archivos
| Archivo | Función |
| --- | --- |
| `config.py` | `LLMSettings.from_env()`: `HEB_LLM_ENABLED`, `OLLAMA_BASE_URL`, `HEB_LLM_MODEL`, `HEB_LLM_TIMEOUT_S`, `HEB_LLM_MAX_OUTPUT_TOKENS` |
| `ollama_client.py` | `POST {OLLAMA_BASE_URL}/api/chat` sin streaming, `temperature=0`, `seed=42`. Errores tipados: no disponible, timeout, respuesta vacía, JSON o HTTP inválidos. Lee los contadores reales `prompt_eval_count` y `eval_count`. |
| `prompts.py` | Mensaje de sistema con las reglas y mensaje de usuario con la respuesta verificada delimitada en `<respuesta_verificada>` como **datos**, más `<contradicciones>` cuando el RAG las detectó |
| `validation.py` | Comprobaciones conservadoras sobre la redacción (ver abajo) |
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

**Costo:** la inferencia es local, así que no hay costo por token de API. Sí hay costo de ejecución local: el modelo ocupa memoria (`load_duration` de Ollama) y CPU o GPU durante segundos por respuesta, frente a milisegundos del modo determinista. Los tokens de embeddings son otro consumo distinto, que se reporta por separado.

## Comparar los dos modos en tu equipo
El script ejecuta el Modo A (determinista) y el Modo B (llama3.2) sobre las mismas 12 preguntas; activa el LLM por sí mismo en el Modo B.
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

**El servidor no mantiene memoria conversacional.** `conversation_id` se acepta por compatibilidad pero no crea una sesión. Streamlit puede enviar explícitamente `previous_question` (una pregunta reciente del usuario, máximo 4,000 caracteres) para resolver seguimientos breves de devoluciones, por ejemplo «Si no tiene el ticket» después de preguntar por devolver un producto. La API no guarda historial ni entrega ese contexto a Ollama: solo se usa para determinar si «ticket» significa comprobante de compra o ticket de soporte. Sin contexto, la pregunta ambigua pide aclaración. Para otros seguimientos, incluye los detalles necesarios en la nueva pregunta.

**Transparencia:** Streamlit muestra un aviso visible cuando la respuesta es conversación general no verificada. `metrics.generated=true` por sí solo **no significa** que la respuesta haya sido contrastada contra datos empresariales. No se deben presentar las respuestas generales como políticas, procedimientos o criterios de Recursos Humanos de H-E-B.
