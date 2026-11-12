# Limitaciones conocidas y pendientes

Estado al 2026-11-12 (`main` en `5c9fcc6`). Todo lo que sigue está verificado o se indica explícitamente que no lo está. El detalle técnico del RAG está en [`../app/rag/EVALUATION.md`](../app/rag/EVALUATION.md).

## 1. Fallos del RAG en el conjunto independiente (3 de 49)
Conjunto escrito y verificado contra los PDF **antes** de los cambios de la fase 3, y ejecutado una sola vez (`tests/rag/fixtures/holdout_eval.json`). Estos fallos **no se corrigieron** para no contaminar el conjunto.

| Caso | Pregunta | Qué responde | Causa |
| --- | --- | --- | --- |
| `h_ident_sin_ticket` | ¿Qué documento debe presentar el cliente para una devolución sin ticket? | Cita PRO-SAC-021 §8.2 y FAQ P19 (fraude), en vez de §5.1 o P16 ("identificación oficial") | Ranking léxico: "devolución", "ticket" y "cliente" cubren por completo varios pasajes |
| `h_planta` | ¿En cuánto tiempo arranca la planta de emergencia? | Cita la fila "Prioridad Crítica" (planta que *no arranca*), en vez de MAN-OPS-007 §7.1 ("treinta segundos") | Coincidencia léxica de "arranca" |
| `hn_nomina` | ¿Qué día se paga la nómina? | Responde con "Definiciones" en lugar de abstenerse | El truncado a 6 letras hace coincidir "nómina" con "valor **nomina**l" |

En los tres casos **las citas existen en el PDF**; lo que falla es la relevancia o la abstención. Resultado global del conjunto independiente: 46/49, con 0 abstenciones falsas.

## 2. Contradicciones e inconsistencias de los documentos
| # | Documentos | Situación | Tratamiento |
| --- | --- | --- | --- |
| 1 | FAQ P15 (p. 4): **15 días** / PRO-SAC-021 §3 (p. 2): **30 días naturales**, electrodomésticos | Detectada | Se muestran ambas. Prevalece PRO-SAC-021 porque la FAQ §1 dice textualmente que *"prevalece siempre la política o el procedimiento vigente al que se hace referencia"* |
| 2 | POL-OPS-014 §6 (p. 4): rebaja de 60 % a "1 día" / §5.1 (p. 3): el perecedero se retira un día antes de la caducidad | **No detectada** | Documentada. El sistema responde "60 %" con su cita y no advierte la tensión. Requiere confirmación del área responsable. |
| 3 | FAQ P13 (p. 4): devolución de producto vendido caducado "sin restricciones" / PRO-SAC-021 §3 (p. 2): perecederos 48 h con defecto de calidad | **No detectada** | Documentada. Además, el extracto de P13 no incluye la oración "procede sin restricciones". |

El sistema **nunca** elige un documento ganador sin una frase de la fuente que lo justifique.

## 3. Embeddings y Ollama: sin pruebas reales
- El adaptador `OllamaEmbedder` y la fusión híbrida RRF están implementados y tienen pruebas unitarias **con un embedder de prueba determinista**. Esas pruebas validan el código, no la calidad de un modelo.
- **No se ha ejecutado ningún modelo de embeddings real.** El entorno de desarrollo no tiene Ollama, y la política de red bloquea `ollama.com` y `registry.ollama.ai` (403).
- Todas las métricas publicadas son de **BM25**. Los comandos para medir el modo híbrido en local están en `app/rag/EVALUATION.md`.

## 4. Defectos encontrados en la verificación de entrega (abiertos)
| Issue | Área | Defecto |
| --- | --- | --- |
| h8hyf2rw9y-afk/HEB#12 | RAG (`app/rag/answering.py`) | En "¿Cuántos días tiene un cliente para devolver una licuadora?" aparece una viñeta de la FAQ P17 (montos de autorización), marcada como parte de la contradicción solo por estar en la misma página que P15 |
| h8hyf2rw9y-afk/HEB#13 | Frontend/API | (1) La línea "⚠️ Contradicción…" se muestra dentro de la última viñeta en Streamlit. (2) "Fuentes (5)" aunque la respuesta cite 1–3 pasajes. (3) El filtro `_POLICY_TERMS` deja fuera preguntas de política válidas ("¿Qué es SIGMA?"). (4) `requirements.txt` raíz vacío. |

## 5. Límites del enfoque léxico (BM25, sin LLM)
- Morfología y sinónimos. Ejemplos: "vender" frente a "venta"; "se cuenta el inventario" frente a "frecuencia de conteo"; "mantenerse" frente a "Mantenimiento".
- Preguntas de dos términos muy genéricas ("¿Cuál es el plazo de devolución?") devuelven varias filas en vez de pedir aclaración. La aclaración solo se pide con un término.
- Sin OCR: las páginas escaneadas se reportan, pero no se indexan. No aplica a los 4 PDF del ejercicio, que tienen texto.
- Solo se detectan tablas con líneas de rejilla.

## 6. Funcionalidad pendiente (no es una limitación del RAG)
- **Consultas analíticas** (ventas, inventario, tickets) desde `/chat`: pendientes de Codex (h8hyf2rw9y-afk/HEB#11, borrador). Hoy `/chat` responde `insufficient_data` con el mensaje "Todavía no está disponible la consulta conversacional de datos estructurados".
- **Costo en tokens**: no hay modelo elegido ni integrado (ver [`TOKEN_COSTS.md`](TOKEN_COSTS.md)).
- **Windows**: las instrucciones están escritas para PowerShell, pero la instalación y el arranque se midieron solo en Linux.

## 7. Fecha límite
El brief fija el **viernes 25 de septiembre de 2026, 23:59 (centro de México)**. La fecha de trabajo de este entorno es 2026-11-12. Hay que confirmar con la reclutadora el plazo vigente antes de enviar.
