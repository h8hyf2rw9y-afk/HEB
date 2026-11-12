# Evaluación RAG — PDF del ejercicio (fase 2)

Corpus: los 4 PDF sintéticos de `candidatos/politicas/` (no se modifican; una prueba verifica sus hashes antes y después de la extracción).

| Documento | Código | Versión | Vigencia (extraída) | Páginas |
| --- | --- | --- | --- | --- |
| `faq_gerentes_de_tienda.pdf` | FAQ-OPS-001 | 1.4 | 2026-01-12 | 7 |
| `manual_apertura_y_cierre_tienda.pdf` | MAN-OPS-007 | 2.5 | 2025-10-01 | 8 |
| `politica_mermas_y_caducidad.pdf` | POL-OPS-014 | 3.2 | 2025-03-01 | 7 |
| `procedimiento_devoluciones.pdf` | PRO-SAC-021 | 4.0 | 2025-06-15 | 7 |

Índice: 245 fragmentos (120 de prosa, 85 filas de tabla, 20 filas de metadatos, 20 de control de cambios). Construcción: ~7 s en este contenedor (la detección de tablas domina).

## Batería
`tests/rag/fixtures/candidatos_eval.json`: 63 casos. Cada caso indica estado esperado, documento/página válidos, sección esperada y frases verificadas a mano en el PDF.

| Categoría | Casos |
| --- | --- |
| devoluciones | 14 |
| mermas | 12 |
| operación | 8 |
| apertura y cierre | 6 |
| FAQ gerentes / historial de documentos | 6 |
| contradicciones | 3 |
| ambiguas | 4 |
| sin respuesta | 10 |

Métricas:
- **hit@5 / MRR:** algún pasaje del documento y página correctos aparece en el top-5.
- **citas con sección correcta:** la cita apunta además a la sección esperada.
- **citas verificables:** cada palabra del pasaje citado existe en esa página del PDF.
- **frases esperadas:** aparecen en la respuesta.
- **abstención:** casos negativos que devuelven `insufficient_data`.
- **abstención falsa:** casos con respuesta que devuelven `insufficient_data`.
- **conflictos:** se detectan cuando existen y se resuelven solo si la fuente lo indica.
- **latencia.**

## Resultados reales — BM25 (contenedor Linux, Python 3.13, PyMuPDF 2.1)

| Métrica | `main` (PR #5) | esta rama |
| --- | --- | --- |
| Casos que pasan | 32 / 63 | **55 / 63** |
| hit@5 | 0.918 | **0.980** |
| MRR | 0.833 | **0.939** |
| Citas con sección correcta | 0.536 | **0.929** |
| Citas verificables en el PDF | 1.000 | **1.000** |
| Frases esperadas en la respuesta | 0.646 | **0.917** |
| Abstención (negativos) | 0.50 | **0.60** |
| Abstención falsa (positivos) | 0.00 | **0.00** |
| Exactitud de estado | 0.873 | **0.937** |
| Conflictos | 0.50 | **1.00** |
| Latencia p50 / p95 por pregunta | 1.3 / 2.5 ms | 1–2 / 2–4 ms |
| Llamadas y tokens de LLM | 0 | 0 |
| Fechas de vigencia extraídas | 0 / 4 | **4 / 4** |

Por categoría (esta rama): ambiguas 4/4, apertura y cierre 6/6, contradicciones 3/3, devoluciones 11/14, FAQ 6/6, mermas 11/12, operación 8/8, sin respuesta 6/10.

La latencia varía entre ejecuciones del contenedor (p50 entre 1 y 2 ms). La columna `main` se obtuvo con el mismo evaluador y la misma batería aplicados al código de `main`.

## Modo híbrido (Ollama)
**No evaluado con un modelo real.** Ollama no está instalado en este contenedor y la política de red bloquea `ollama.com` y `registry.ollama.ai` (403 del proxy). La ruta híbrida (RRF) solo está cubierta por pruebas unitarias con un embedder determinista de prueba; esas cifras **no** miden la calidad de un modelo de embeddings y no se reportan como resultados híbridos.

Para medirlo en local:
```bash
ollama pull bge-m3
python -m app.rag build --pdf-dir candidatos/politicas --index-dir data/rag_index --embeddings required
python -m app.rag eval --cases tests/rag/fixtures/candidatos_eval.json --index-dir data/rag_index --verify-pdf-dir candidatos/politicas
HEB_RAG_OLLAMA_TESTS=1 python -m pytest tests/rag -q
```

## Contradicciones encontradas en los documentos
1. **Plazo de electrodomésticos y electrónicos menores.** La FAQ P15 (p. 4) da **15 días** con ticket. PRO-SAC-021 §3 (p. 2) da **30 días naturales**. Se detecta automáticamente y se resuelve a favor de PRO-SAC-021 solo porque la FAQ §1 (p. 1) dice: *"en caso de duda o de diferencia con un documento normativo, prevalece siempre la política o el procedimiento vigente al que se hace referencia"*, y la FAQ cita PRO-SAC-021. Ambas evidencias se muestran siempre.
2. **Posible inconsistencia interna de POL-OPS-014** (revisión manual, no detectada). La tabla de rebajas (§6, p. 4) asigna 60 % a "1 día (solo perecederos no lácteos)", pero §5.1 (p. 3) retira el perecedero del anaquel un día antes de la caducidad: el último día de venta es dos días antes. El detector no la marca porque no hay valores distintos para el mismo atributo. Caso `conf_rebaja_un_dia`, marcado como limitación conocida.
3. **Posible diferencia de alcance** (revisión manual, no detectada). La FAQ P13 dice que la devolución de un producto vendido caducado "procede sin restricciones", mientras que PRO-SAC-021 §3 limita los perecederos a 48 horas con defecto de calidad. No hay cifras contrapuestas en la misma frase, así que el detector no la identifica.

## Limitaciones conocidas (casos marcados `known_limitation`)
| Caso | Qué pasa | Causa |
| --- | --- | --- |
| `dev_perecederos` | La fila "Perecederos \| 48 horas" queda fuera del top-5 | Ranking BM25: "tiempo"/"devolver" favorecen 3.1 y 7.1 |
| `dev_sin_ticket` | Cita FAQ P16, pero el extracto no incluye "vale de tienda" | Puntuaciones casi empatadas; el extracto principal sale de §4.1 |
| `dev_vale_vigencia` | Cita "2. Definiciones", pero no la oración "doce meses" | Selección de oración |
| `mer_conteo_abarrotes` | Se filtra la fila "Abarrotes" (cobertura 0.4475 < 0.45) | Vocabulario "se cuenta el inventario" frente a "Frecuencia de conteo" |
| `conf_rebaja_un_dia` | No se marca la posible inconsistencia | Ver contradicción 2 |
| `neg_estacionamiento` | Responde con la fila 21:30 del cierre en vez de abstenerse (cobertura 0.4998) | Cobertura léxica en el límite |
| `neg_tolerancia_electronica` | Devuelve filas de otras categorías | La cobertura de "tolerancia" + "merma" basta; no hay comprobación de entidad |
| `neg_ventas_agosto` | Responde con FAQ P13 ("vendió") | Pregunta de datos estructurados: debe enrutarla el backend a DuckDB antes del RAG |
| `neg_precio_leche` | Responde con la regla de retiro de lácteos | "leche" es un término raro; la cobertura léxica no distingue la intención |

En todos estos casos la respuesta cita pasajes reales y verificables; no se inventan cifras. Lo que falla es la relevancia o la abstención. La prueba `test_evaluation_battery` exige que todo caso **no** marcado como limitación pase, y que se mantengan las citas verificables al 100 %, 0 abstenciones falsas y conflictos al 100 %.

## Sensibilidad del umbral de cobertura (medida, no aplicada)
| `HEB_RAG_MIN_COVERAGE` | Casos que pasan | Abstención | Abstención falsa | Negativos que aún responden |
| --- | --- | --- | --- | --- |
| **0.45 (actual)** | 55 / 63 | 0.60 | 0.00 | estacionamiento, tolerancia electrónica, ventas agosto, precio leche |
| 0.50 | 56 / 63 | 0.80 | 0.00 | tolerancia electrónica, precio leche |
| 0.55 | 55 / 63 | 0.80 | 0.00 | tolerancia electrónica, precio leche |

En esta batería, 0.50 sería mejor que 0.45 y no causaría abstenciones falsas. **No se cambió**: el criterio de la tarea es no ajustar umbrales para maximizar métricas sobre un conjunto pequeño. Además, los dos negativos que se corregirían están en el límite (0.4998 y 0.4847) y los preguntó la misma persona que diseñó el sistema. Recomendación: decidir el umbral con un conjunto de preguntas reales de gerentes que no se haya usado para desarrollar, o tras medir el modo híbrido con Ollama. Se puede probar sin cambiar código con `HEB_RAG_MIN_COVERAGE=0.5`.
