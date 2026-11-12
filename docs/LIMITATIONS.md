# Limitaciones conocidas

Estado al 2026-11-12 (`main` en `020c23c`). Todo lo que sigue está verificado o se indica explícitamente que no lo está. Detalle del RAG en [`../app/rag/EVALUATION.md`](../app/rag/EVALUATION.md).

## 1. Alcance de las consultas de datos
`app/tools/analytics.py` responde un **catálogo acotado de intenciones** con SQL fijo:

- **Ventas netas** (unidades o descuentos) por mes de junio a agosto de 2026 o del periodo completo: total, por tienda, o ranking por tienda, producto, categoría o canal.
- **Inventario** al corte del 2026-09-01: productos en o por debajo del punto de reorden, y agotados.
- **Tickets:** conteos por prioridad y estado, por tienda.

No cubre (y debe abstenerse): precios, costos, márgenes, pronósticos, comparaciones entre varios meses, análisis del texto de los tickets ni tiempos de resolución. Defectos observados dentro del alcance:

| Pregunta | Qué responde | Problema |
| --- | --- | --- |
| ¿Cuántas unidades hay en el inventario de la tienda T03? | "Tienda Anáhuac: 5 SKU en o por debajo del punto de reorden…" | No responde unidades; da un resumen de reorden |
| ¿Cuántas unidades hay en el inventario actual de la tienda 3? | El mismo resumen, **para las 4 tiendas** | "tienda 3" no se reconoce como T03 |
| ¿Qué productos están agotados? | `insufficient_data`: "…No hay productos que cumplan la condición **para esa tienda**." | El dato es correcto (no hay existencias en 0), pero el mensaje habla de una tienda que no se preguntó |

## 2. Enrutamiento entre datos y políticas (reglas léxicas)
`/chat` decide con `looks_like_data`: lo que parece de datos va a DuckDB y el resto al RAG. Al pasar las **112 preguntas de política** de las dos baterías del RAG por ese enrutamiento, **3 se desvían a datos y reciben cifras reales que no responden lo preguntado**:

| Pregunta (política) | Respuesta recibida por `/chat` |
| --- | --- |
| ¿Cuántas devoluciones sin ticket puede hacer un mismo cliente en doce meses? | "Tickets registrados, por tienda…" (confunde el ticket de compra con el de la mesa de servicio) |
| ¿Con qué frecuencia se cuenta el inventario de abarrotes? | "Inventario al 1 de septiembre de 2026: … SKU en o por debajo del punto de reorden…" |
| ¿Cuánto tiempo puede permanecer en tienda un producto mermado después de su registro? | "Ventas netas por producto (junio–agosto de 2026)…" |

Llamando al RAG directamente, las tres se responden bien. Las baterías del RAG prueban el módulo directamente, no `/chat`, así que el CI no lo detectó. Se reportó en h8hyf2rw9y-afk/HEB#13 con propuestas de corrección para Codex.

## 3. Fallos del RAG en el conjunto independiente (3 de 49)
Conjunto escrito y verificado contra los PDF antes de los cambios de la fase 3, y ejecutado una sola vez. No se corrigieron para no contaminarlo.

| Caso | Pregunta | Qué responde | Causa |
| --- | --- | --- | --- |
| `h_ident_sin_ticket` | ¿Qué documento debe presentar el cliente para una devolución sin ticket? | Cita PRO-SAC-021 §8.2 y FAQ P19 (fraude), no §5.1 ni P16 ("identificación oficial") | Ranking léxico |
| `h_planta` | ¿En cuánto tiempo arranca la planta de emergencia? | Cita la fila "Prioridad Crítica" (planta que *no arranca*), no MAN-OPS-007 §7.1 ("treinta segundos") | Coincidencia léxica de "arranca" |
| `hn_nomina` | ¿Qué día se paga la nómina? | Responde con "Definiciones" en lugar de abstenerse | Truncado a 6 letras: "nómina" ≈ "valor **nomina**l" |

En los tres casos **las citas existen en el PDF**; lo que falla es la relevancia o la abstención. Resultado global: 46/49, con 0 abstenciones falsas.

## 4. Contradicciones e inconsistencias de los documentos
| # | Documentos | Situación | Tratamiento |
| --- | --- | --- | --- |
| 1 | FAQ P15 (p. 4): **15 días** / PRO-SAC-021 §3 (p. 2): **30 días naturales**, electrodomésticos | Detectada | Se muestran ambas. Prevalece PRO-SAC-021 porque la FAQ §1 dice que *"prevalece siempre la política o el procedimiento vigente al que se hace referencia"* |
| 2 | POL-OPS-014 §6 (p. 4): rebaja de 60 % a "1 día" / §5.1 (p. 3): el perecedero se retira un día antes de la caducidad | **No detectada** | Documentada; el sistema responde "60 %" con su cita y no advierte la tensión |
| 3 | FAQ P13 (p. 4): devolución de producto vendido caducado "sin restricciones" / PRO-SAC-021 §3 (p. 2): perecederos 48 h con defecto de calidad | **No detectada** | Documentada; además, el extracto de P13 no incluye la oración "procede sin restricciones" |

El sistema **nunca** elige un documento ganador sin una frase de la fuente que lo justifique.

## 5. Defectos abiertos
| Issue | Estado | Detalle |
| --- | --- | --- |
| h8hyf2rw9y-afk/HEB#12 (RAG) | **Abierto** | En la pregunta de la licuadora aparece una viñeta de la FAQ P17 (montos de autorización): entra como parte de la contradicción solo por estar en la misma página que P15. La corrección (comparar por oración, no por página) está probada y sin regresiones en las dos baterías; falta autorizar el cambio en `app/rag`. |
| h8hyf2rw9y-afk/HEB#13 (frontend/API) | Puntos originales **resueltos** por h8hyf2rw9y-afk/HEB#11; quedan los nuevos | Resuelto: ⚠️ separado de la viñeta, "Fuentes" solo con lo citado, "¿Qué es SIGMA?" llega al RAG, `requirements.txt` completo. Nuevo: errores de enrutamiento (§2) y de alcance (§1). |

## 6. Embeddings y Ollama: sin pruebas reales
- El adaptador `OllamaEmbedder` y la fusión híbrida RRF tienen pruebas unitarias **con un embedder de prueba**, que valida el código, no la calidad de un modelo.
- **No se ha ejecutado ningún modelo de embeddings real.** El entorno de desarrollo no tiene Ollama y la red bloquea `ollama.com` (403).
- Todas las métricas de políticas son de **BM25**.

## 7. Límites del enfoque léxico (BM25, sin LLM)
- Morfología y sinónimos: "vender" frente a "venta", "se cuenta" frente a "conteo", "mantenerse" frente a "Mantenimiento".
- Preguntas genéricas de dos términos ("¿Cuál es el plazo de devolución?") listan varias filas en vez de pedir aclaración.
- Respuestas extractivas: precisas, pero no redactadas.
- Sin OCR; solo se detectan tablas con líneas de rejilla. No afecta a los 4 PDF del ejercicio.

## 8. Plataforma y entrega
- **Windows:** la instalación y las pruebas pasan en GitHub Actions con Python 3.11.9 (instalación 62 s, pruebas 22 s). El arranque completo de API + Streamlit y `verify_delivery.py` **no se han ejecutado en un Windows real**.
- **Fecha límite:** el brief fija el 25 de septiembre de 2026, 23:59 (centro de México). La fecha de trabajo de este entorno es 2026-11-12; hay que confirmar el plazo.
