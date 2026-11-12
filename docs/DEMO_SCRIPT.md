# Guion de demostración (3–5 minutos)

Para el video que pide el brief (*"3 a 5 minutos mostrando el sistema funcionando. Sin edición ni presentación formal"*).

- Las respuestas de **políticas** citadas aquí son salidas reales de `/chat` (entorno limpio, `main` en `5c9fcc6`, 2026-11-12).
- Las **consultas analíticas** son espacios reservados: se rellenan **solo** con lo que devuelva el sistema cuando Codex entregue DuckDB (h8hyf2rw9y-afk/HEB#11). No grabar esa parte antes.

## Preparación (antes de grabar, ~2 min)
1. Instalar según [`../README.md`](../README.md) y reconstruir el índice:
   ```powershell
   .\.venv\Scripts\python.exe -m app.rag build --pdf-dir candidatos/politicas --index-dir data/rag_index --embeddings off
   ```
2. Comprobar que todo responde (debe terminar con `fallos: 0`):
   ```powershell
   .\.venv\Scripts\python.exe scripts\verify_delivery.py
   ```
3. Terminal 1, API:
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.api.main:app --port 8000
   ```
4. Terminal 2, interfaz:
   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run frontend/app.py
   ```
5. En el navegador: barra lateral en "API conectada" y pulsar **Nueva conversación**.
6. Tener a mano este guion y el PDF `procedimiento_devoluciones.pdf`, abierto en la página 2, para mostrar que la cita es real.

## Guion

| Tiempo | Qué mostrar | Qué decir (resumen) |
| --- | --- | --- |
| 0:00–0:30 | Pantalla del chat | "Una gerente con cuatro tiendas quiere preguntar en español por sus tiendas y sus políticas, y que el sistema le diga cuando no sabe. Todo corre en local: Streamlit, FastAPI, DuckDB para los datos y un buscador sobre los PDF que siempre cita documento, página y sección." |
| 0:30–1:15 | **P1 · contradicción** | Ver P1. Mostrar el aviso ⚠️ y abrir el PDF en la página 2 para comparar. |
| 1:15–1:45 | **P2 · restricciones completas** | Ver P2. |
| 1:45–2:05 | **P3 · tabla** | Ver P3. |
| 2:05–2:30 | **P4 · no inventa** | Ver P4. "Prefiere decir que no sabe a dar una cifra de otra categoría." |
| 2:30–3:45 | **A1–A3 · analíticas** ⏳ | Solo con resultados reales de DuckDB (ver más abajo). |
| 3:45–4:20 | "Cómo sé que funciona" | Mostrar `app/rag/EVALUATION.md`: 49 preguntas independientes, 46 correctas, 0 abstenciones falsas, citas verificadas palabra por palabra contra el PDF. Mencionar que hay una batería equivalente para las analíticas ⏳. |
| 4:20–4:45 | Costo | Mostrar el desplegable "Métricas de la consulta". **Solo** cifras medidas (ver [`TOKEN_COSTS.md`](TOKEN_COSTS.md)); hoy: 0 llamadas LLM y 0 tokens en políticas. ⏳ Costo con el modelo elegido. |
| 4:45–5:00 | Limitaciones | "No detecta todas las inconsistencias entre documentos (hay dos documentadas), no tiene embeddings probados todavía y falla 3 de 49 preguntas independientes; están documentadas." |

### P1 — Contradicción entre documentos
**Pregunta:** `¿Cuántos días tiene un cliente para devolver una licuadora?`

Salida real (extracto):
```
Respuesta según los documentos:
- Tipo de producto: Electrodomésticos y electrónicos menores | Plazo para devolución o cambio: 30 días naturales | Requisitos: Ticket de compra, empaque original completo con accesorios y manuales. [procedimiento_devoluciones.pdf, p. 2, sección: 3. Plazos de devolución por tipo de producto]
- Para electrodomésticos y electrónicos menores como licuadoras, ... el cliente tiene 15 días con ticket ... [faq_gerentes_de_tienda.pdf, p. 4, sección: P15. ...] (no prevalece según la fuente; ver contradicción)
⚠️ Contradicción entre documentos (días): faq_gerentes_de_tienda.pdf p. 4: 15; procedimiento_devoluciones.pdf p. 2: 30. Prevalece procedimiento_devoluciones.pdf. faq_gerentes_de_tienda.pdf indica: "... prevalece siempre la política o el procedimiento vigente al que se hace referencia." ...
```
Qué decir: "Los dos documentos dicen cosas distintas. El sistema muestra ambos y solo da prioridad al procedimiento porque la propia FAQ dice que el procedimiento prevalece; si no lo dijera, lo marcaría como sin resolver."

> ⚠️ **Mientras h8hyf2rw9y-afk/HEB#12 siga abierto**, la respuesta incluye además una viñeta de la FAQ P17 (montos de autorización) que no viene al caso, y en Streamlit la línea ⚠️ aparece pegada a esa viñeta (h8hyf2rw9y-afk/HEB#13). Si no se ha corregido al grabar, dilo en una frase ("esta viñeta sobra; es un defecto conocido") o usa P1-alternativa.

**P1-alternativa** (sin contradicción, mismo documento): `¿Cuál es el plazo para devolver abarrotes?` → `Tipo de producto: Abarrotes y no perecederos | Plazo para devolución o cambio: 30 días | ...` `[procedimiento_devoluciones.pdf, p. 2, ...]`

### P2 — Respuesta directa con todas las restricciones
**Pregunta:** `¿Cómo se hace la devolución si el cliente pagó con tarjeta y no trae la tarjeta original?`

Salida real:
```
Respuesta según los documentos:
- Si el cliente pagó con tarjeta y no la presenta al momento de la devolución, el reembolso se realiza en vale de tienda. No se realizan reembolsos en efectivo de compras pagadas con tarjeta bajo ninguna circunstancia, ni reembolsos a una tarjeta distinta de la original. [procedimiento_devoluciones.pdf, p. 4, sección: 6.1 Cliente sin la tarjeta original]
```
Qué decir: "Responde lo que se preguntó, incluye las dos prohibiciones y no añade las demás formas de pago."

### P3 — Dato de una tabla del PDF
**Pregunta:** `¿Cuál es la tolerancia de merma en panadería?`

Salida real:
```
- Categoría: Panadería | Tolerancia mensual (% de venta): 3.5 | Umbral de alerta (%): 4.5 | Frecuencia de conteo: Diaria. [politica_mermas_y_caducidad.pdf, p. 3, sección: 4. Tabla de tolerancias de merma por categoría]
```

### P4 — Reconoce lo que no sabe
**Pregunta:** `¿Cuál es la tolerancia de merma para electrónica?`

Salida real (`insufficient_data`):
```
No encontré información suficiente en los documentos de políticas disponibles para responder esta pregunta. ...
ℹ️ La tabla «4. Tabla de tolerancias de merma por categoría» (politica_mermas_y_caducidad.pdf, p. 3) no incluye «electrónica»; incluye: Lácteos, Panadería, Carnes, Abarrotes, Bebidas, Limpieza, Frutas y verduras, Cuidado personal.
```

### A1–A3 — Consultas analíticas (⏳ pendiente de Codex)
**No grabar ni rellenar hasta que la funcionalidad esté fusionada en `main`.** Hoy `/chat` responde a estas preguntas con `insufficient_data` y el texto "Todavía no está disponible la consulta conversacional de datos estructurados…".

| # | Pregunta sugerida | Qué debe mostrar el video | Resultado real | Evidencia esperada |
| --- | --- | --- | --- | --- |
| A1 | ¿Qué tienda vendió más en agosto de 2026? | Tienda, venta neta en MXN y periodo (2026-08-01 a 2026-08-31); devoluciones (unidades negativas) netas | `[PENDIENTE: salida real de /chat]` | `ventas.csv`, periodo |
| A2 | ¿Qué productos de la tienda T01 están por debajo del punto de reorden? | Lista SKU y nombre, existencia frente a punto de reorden, **corte 2026-09-01** | `[PENDIENTE]` | `inventario.csv` + `catalogo_productos.csv`, fecha de corte |
| A3 | ¿Cuántos tickets de prioridad Crítica siguen abiertos y de qué tiendas son? | Conteo por tienda, estado al corte | `[PENDIENTE]` | `tickets_mesa_servicio.csv` |
| A4 (opcional) | ¿Cuánto vamos a vender en diciembre? | Debe **abstenerse**: no hay datos de diciembre ni pronósticos | `[PENDIENTE]` | — |

Cifras que conviene que alguien verifique a mano (por ejemplo con DuckDB o una hoja de cálculo) antes de grabar: el total de A1 y el conteo de A3.

## Si algo falla durante la grabación
- "Sin conexión" en la barra lateral: la API no está activa; revisar la terminal 1.
- `Índice de políticas no disponible`: falta el paso 1 de la preparación.
- Errores de codificación en la consola de Windows: ejecutar `$env:PYTHONUTF8 = "1"` en esa terminal y volver a lanzar.
