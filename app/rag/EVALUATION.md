# Evaluación RAG — PDF del ejercicio (fases 2 y 3)

Corpus: los 4 PDF sintéticos de `candidatos/politicas/`. No se modifican; una prueba compara sus hashes antes y después de la extracción.

| Documento | Código | Versión | Vigencia (extraída) | Páginas |
| --- | --- | --- | --- | --- |
| `faq_gerentes_de_tienda.pdf` | FAQ-OPS-001 | 1.4 | 2026-01-12 | 7 |
| `manual_apertura_y_cierre_tienda.pdf` | MAN-OPS-007 | 2.5 | 2025-10-01 | 8 |
| `politica_mermas_y_caducidad.pdf` | POL-OPS-014 | 3.2 | 2025-03-01 | 7 |
| `procedimiento_devoluciones.pdf` | PRO-SAC-021 | 4.0 | 2025-06-15 | 7 |

Índice: 245 fragmentos (120 de prosa, 85 filas de tabla, 20 de metadatos, 20 de control de cambios). Construcción: ~7 s.

## Conjuntos de preguntas
| Conjunto | Archivo | Casos | Uso |
| --- | --- | --- | --- |
| Desarrollo | `tests/rag/fixtures/candidatos_eval.json` | 63 (49 con respuesta, 4 ambiguas, 10 sin respuesta) | Diagnóstico y desarrollo de las fases 2 y 3 |
| **Independiente** | `tests/rag/fixtures/holdout_eval.json` | 49 (32 con respuesta, 2 ambiguas, 15 sin respuesta) | Escrito y verificado contra los PDF **antes** de cualquier cambio de la fase 3 (commit `ecdc07d`). No se usó para ajustar nada; se ejecutó una vez al final. |

Cada caso indica:
- **estado esperado;**
- **documento y página válidos;**
- **sección y frases esperadas.** Se comprobó con un script que cada frase existe en la página indicada del PDF.

Métricas:
- **hit@5 / MRR:** pasaje del documento y página correctos en el top-5.
- **citas con sección:** además, la sección es la correcta.
- **citas verificables:** cada palabra del pasaje citado existe en esa página del PDF.
- **frases:** las frases esperadas aparecen en la respuesta.
- **abstención correcta:** negativos que devuelven `insufficient_data`.
- **abstención falsa:** casos con respuesta que devuelven `insufficient_data`.
- **conflictos.**
- **latencia.**

## Resultados reales — BM25 (contenedor Linux, Python 3.13, PyMuPDF 2.1)

### Desarrollo (63)
| Métrica | `main` (PR #5) | PR #10 (fase 2) | **fase 3** |
| --- | --- | --- | --- |
| Casos que pasan | 32 | 55 | **63** |
| hit@5 / MRR | 0.918 / 0.833 | 0.980 / 0.939 | **1.000 / 0.980** |
| Citas con sección correcta | 0.536 | 0.929 | **1.000** |
| Citas verificables | 1.000 | 1.000 | **1.000** |
| Frases esperadas | 0.646 | 0.917 | **1.000** |
| Abstención correcta | 0.50 | 0.60 | **1.00** |
| Abstención falsa | 0.00 | 0.00 | **0.00** |
| Conflictos | 0.50 | 1.00 | **1.00** |
| Latencia p50 / p95 | ~1.3 / 2.5 ms | ~1–2 / 2–4 ms | ~2–3 / 4–7 ms |

Llamadas y tokens de LLM: 0 en todas las versiones.

### Independiente (49) — la medida que importa
| Métrica | PR #10 (antes) | **fase 3** |
| --- | --- | --- |
| Casos que pasan | 45 | **46** |
| hit@5 | 0.969 | **1.000** |
| Citas verificables | 1.000 | **1.000** |
| Frases esperadas | 0.938 | **0.938** |
| Abstención correcta | 0.867 | **0.933** |
| Abstención falsa | 0.031 | **0.000** |
| Exactitud de estado | 0.939 | **0.980** |

El 63/63 de desarrollo **no** es una cifra de generalización: los cambios se desarrollaron mirando esas preguntas. La cifra realista es la del conjunto independiente, **46/49**.

## Umbral de abstención (`HEB_RAG_MIN_COVERAGE`)
| Umbral | Desarrollo | Independiente | Abstención correcta (indep.) | Abstención falsa (indep.) |
| --- | --- | --- | --- | --- |
| 0.40 | — | 44 / 49 | 0.800 | 0.000 |
| **0.45 (actual)** | 63 / 63 | **46 / 49** | **0.933** | **0.000** |
| 0.50 | 63 / 63 | 46 / 49 | 0.933 | 0.000 |
| 0.55 | — | 46 / 49 | 0.933 | 0.000 |

Antes de la fase 3, con el código del PR #10, 0.45 y 0.50 también daban lo mismo en el conjunto independiente (45/49).

**Recomendación: mantener 0.45.** En los dos conjuntos 0.45 y 0.50 son idénticos, así que no hay evidencia para cambiarlo. Bajar a 0.40 empeora la abstención (0.93 → 0.80). La mejora de abstención de la fase 3 no viene del umbral sino de reglas explícitas: foco de la pregunta, entidad ausente en una tabla y preguntas de cifras de ventas. Si hay que elegir entre 0.45 y 0.50 en el futuro, hace falta un conjunto con más negativos cercanos al límite.

## Modo híbrido (Ollama)
**No evaluado con un modelo real.** Ollama no está instalado en este contenedor y la política de red bloquea `ollama.com` y `registry.ollama.ai` (403 del proxy). La ruta RRF solo se prueba con un embedder determinista de prueba, y esas pruebas **no** miden la calidad de un modelo. Comandos para medirlo en local:
```bash
ollama pull bge-m3
python -m app.rag build --pdf-dir candidatos/politicas --index-dir data/rag_index --embeddings required
python -m app.rag eval --cases tests/rag/fixtures/holdout_eval.json --index-dir data/rag_index --verify-pdf-dir candidatos/politicas
HEB_RAG_OLLAMA_TESTS=1 python -m pytest tests/rag -q
```

## Qué cambió en la fase 3 (y qué caso corrige)
| Regla (general, documentada en `answering.py`) | Casos de desarrollo corregidos |
| --- | --- |
| **Reordenar por respuesta directa:** cobertura + coincidencia con el título de sección (peso 0.5) + entidad de fila de tabla | `dev_perecederos`, `mer_conteo_abarrotes` |
| **Tabla como entidad + atributo:** una fila es relevante si su etiqueta (o el valor de otra columna) coincide con la entidad preguntada y un encabezado con el atributo, aunque la cobertura sea baja. Las filas hermanas de otras entidades se descartan. | `dev_perecederos`, `mer_conteo_abarrotes`, "3 días", "monto del Gerente de Tienda" |
| **Entidad ausente en una tabla:** si coinciden los encabezados pero ninguna fila es la entidad preguntada, se responde `insufficient_data` y se listan las filas que la tabla sí contiene | `neg_tolerancia_electronica` (y, sin haberlo visto antes, `hn_tol_farmacia` y `hn_llantas` del conjunto independiente) |
| **Foco de la pregunta:** en "¿Cuál es el *X* de…?", "¿Cuánto cuesta…?" o "¿Qué precio…?", X debe aparecer en la evidencia | `neg_precio_leche`, `neg_estacionamiento` |
| **Preguntas de cifras de ventas** (verbo de venta o "ventas" + cantidad o ranking): abstención y aviso de que corresponden a datos estructurados | `neg_ventas_agosto` |
| **Precedencia normativa al elegir la cita principal:** si una FAQ que se declara subordinada y una política o procedimiento empatan (≥ 90 %), cita primero el normativo | `mer_conteo_abarrotes` |
| **Extracto:** oraciones con ≥ 60 % de la puntuación y ≥ 50 % de los términos de la mejor; restricciones que comparten 2 términos; la entrada de FAQ cuyo título es la pregunta se cita completa salvo las referencias cruzadas; la oración que define el término ("Vale de tienda: …") pesa doble; el foco siempre aparece | `dev_sin_ticket`, `dev_vale_vigencia`, el ejemplo "sin tarjeta original" |
| **Pasajes secundarios:** solo si están en un conflicto, o casi empatan (≥ 95 %) en la misma sección, o corroboran desde otro documento (Jaccard ≥ 0.2 fuera de los términos de la pregunta), y en todos los casos con todos los términos que tiene el principal | Elimina las filas de débito y crédito en "sin tarjeta" y la FAQ de bitácora en "alarma" |
| **Determinismo:** las sumas de flotantes se hacen en orden fijo; antes el resultado dependía de `PYTHONHASHSEED` | Una prueba compara las respuestas con dos semillas |
| **Palabras vacías:** se añaden preposiciones ("durante", "mediante", "hacia", "tras") | `ap_cajas_cierre` |

**Probado y descartado** porque empeoraba el conjunto de desarrollo:
- regla de diptongación (*mantienen → mantenen*): arreglaba 1 caso y rompía 2;
- mezcla de la puntuación BM25 en el reordenamiento: se probaron 0.3, 0.5, 0.7, 0.85 y 1.0;
- peso 0.3 del título de sección.

## Respuesta: estructura
```
Respuesta según los documentos:
- <oraciones copiadas del pasaje> [archivo.pdf, p. N, sección: …]
- <pasaje que corrobora o que está en conflicto> [ … ] (no prevalece según la fuente; ver contradicción)
⚠️ Contradicción entre documentos (…): … Prevalece … <frase de la fuente>   |   ⚠️ … sin resolver …
ℹ️ Los documentos no mencionan: <términos que no existen en el corpus>; la respuesta cubre solo lo demás.
```
En `insufficient_data` el texto explica el motivo cuando se conoce:
- **atributo no encontrado:** «Los documentos no indican «precio»…»;
- **entidad ausente en la tabla:** «La tabla … no incluye «electrónica»; incluye: …»;
- **consulta de ventas:** «Parece una consulta de datos de ventas…».

En ese estado `evidence` sigue vacío, igual que en la fase 2.

## Contradicciones e inconsistencias de los documentos
1. **Detectada y resuelta por la fuente: plazo de electrodomésticos.**
   - FAQ P15 (p. 4): 15 días.
   - PRO-SAC-021 §3 (p. 2): 30 días naturales.
   - Prevalece PRO-SAC-021, porque la FAQ §1 dice que *"prevalece siempre la política o el procedimiento vigente al que se hace referencia"* y la FAQ lo cita.
   - Ambas evidencias se muestran siempre.
2. **No detectada (documentada): posible inconsistencia interna de POL-OPS-014.**
   - La tabla de rebajas (§6, p. 4) da 60 % a "1 día (solo perecederos no lácteos)".
   - §5.1 (p. 3) retira el perecedero del anaquel un día antes de la caducidad (último día de venta: dos días antes).
   - El detector solo compara valores distintos para el mismo atributo; aquí hace falta razonar sobre fechas. El sistema responde 60 % con su cita, sin advertir la tensión.
3. **No detectada (documentada): posible diferencia de alcance FAQ P13 / PRO-SAC-021.**
   - FAQ P13 (p. 4): la devolución de un producto vendido caducado "procede sin restricciones".
   - PRO-SAC-021 §3 (p. 2): limita los perecederos a 48 horas, con defecto de calidad (incluida "caducidad vencida al momento de la compra").
   - No hay cifras contrapuestas en frases comparables. Además, para "¿Se puede devolver un producto que se vendió ya caducado?" el extracto de P13 no incluye la oración "procede sin restricciones": comparte un solo término con la pregunta.

Ambas inconsistencias requieren confirmación del área responsable; el sistema no las resuelve.

## Fallos que continúan
**Conjunto independiente (3 de 49), sin corregir para no contaminar el conjunto:**

| Caso | Qué pasa | Causa |
| --- | --- | --- |
| `h_ident_sin_ticket` | Cita PRO-SAC-021 §8.2 y FAQ P19 (fraude) en lugar de §5.1 o P16 ("identificación oficial") | Ranking: "devolución", "ticket" y "cliente" cubren 100 % en varios pasajes; "documento" y "presentar" no discriminan |
| `h_planta` | Cita la fila "Prioridad Crítica" (planta que *no arranca*) en lugar de MAN §7.1 ("treinta segundos") | Coincidencia léxica de "arranca" |
| `hn_nomina` | Responde con "Definiciones" en lugar de abstenerse | El truncado a 6 letras hace coincidir "nómina" con "valor **nomina**l" |

**Observados fuera de los conjuntos, al probar a mano:**
- "¿Hasta cuándo se puede vender un perecedero?" responde con la fila de devoluciones de perecederos. "vender" no coincide con "venta" en §5.1.
- En "cámaras durante una falla de energía" el extracto incluye oraciones de la misma contingencia que no son estrictamente la respuesta.

Estos casos apuntan a límites del enfoque léxico (morfología y sinónimos). Embeddings o un reordenador semántico deberían cubrirlos; hay que medirlos con Ollama real.

## Por qué el resultado es confiable (y dónde no)
- **Cada oración de una respuesta se copia de un pasaje citado.** Las citas se verifican palabra por palabra contra la página del PDF (100 % en los 112 casos).
- **0 abstenciones falsas en los dos conjuntos.** Cuando el sistema no tiene evidencia, lo dice y explica el motivo concreto.
- **Los conflictos se muestran siempre con ambas partes.** Solo se elige una cuando la fuente lo dice.
- **No es infalible:** en el conjunto independiente, 2 respuestas citan pasajes reales pero no el que contiene el dato, y 1 pregunta sin respaldo se responde con un pasaje irrelevante. El orquestador o la interfaz deben mostrar siempre las citas, para que el usuario pueda verificarlas.
