# Guion de demostración (3–5 minutos)

Para el video que pide el brief (*"3 a 5 minutos mostrando el sistema funcionando. Sin edición ni presentación formal"*).

Todas las respuestas citadas son **salidas reales de `/chat`** (`main` en `020c23c`, entorno limpio, 2026-11-12). Las cifras de datos se verificaron además por separado, leyendo los CSV con Python (`csv` + `Decimal`, sin DuckDB); `scripts/verify_delivery.py` repite esa verificación.

## Preparación (antes de grabar, ~2 min)
1. Instalar según [`../README.md`](../README.md) (`pip install -r requirements.txt`) y reconstruir el índice:
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
6. Tener abierto `procedimiento_devoluciones.pdf` en la página 2, para mostrar que la cita es real.

## Guion

| Tiempo | Qué mostrar | Qué decir (resumen) |
| --- | --- | --- |
| 0:00–0:25 | Pantalla del chat | "Una gerente con cuatro tiendas quiere preguntar en español por sus ventas, su inventario, sus tickets y sus políticas, y que el sistema le diga cuando no sabe. Todo corre en local: Streamlit, FastAPI, DuckDB sobre los CSV y un buscador sobre los PDF que cita documento, página y sección." |
| 0:25–1:00 | **A1 · ventas** | Ver A1. Abrir "Fuentes" (`ventas.csv` · agosto de 2026). |
| 1:00–1:30 | **A2 · inventario** | Ver A2. Remarcar la fecha de corte. |
| 1:30–1:50 | **A3 · tickets** | Ver A3. |
| 1:50–2:05 | **A4 · no inventa cifras** | Ver A4. |
| 2:05–2:50 | **P1 · contradicción entre políticas** | Ver P1. Mostrar el aviso ⚠️ y el PDF en la página 2. |
| 2:50–3:15 | **P2 · restricciones completas** | Ver P2. |
| 3:15–3:35 | **P4 · no inventa políticas** | Ver P4. |
| 3:35–4:15 | "Cómo sé que funciona" | Políticas: 49 preguntas independientes, 46 correctas, 0 abstenciones falsas y citas verificadas palabra por palabra contra el PDF (`app/rag/EVALUATION.md`). Datos: las cifras de la demo se recalculan sin DuckDB en las pruebas, en Windows y Linux. |
| 4:15–4:40 | Costo | Desplegable "Métricas de la consulta": 0 llamadas a LLM y 0 tokens. "Decidí no usar un modelo generativo: las cifras salen de SQL y las políticas se citan textualmente, así que el costo por interacción en tokens es cero." (ver [`TOKEN_COSTS.md`](TOKEN_COSTS.md)). |
| 4:40–5:00 | Limitaciones | "Responde un catálogo acotado de preguntas de datos; el enrutamiento por reglas a veces confunde datos y políticas; falla 3 de 49 preguntas independientes de políticas y no detecta dos inconsistencias entre documentos; todo está documentado." |

P3 (tabla de un PDF) es opcional si sobra tiempo.

### A1 — Ventas por tienda
**Pregunta:** `¿Qué tienda vendió más en agosto de 2026?`
```
Ventas netas por tienda (agosto de 2026), mayor a menor:
1. Tienda Valle Norte: $2,163,062.36 MXN
2. Tienda Cumbres: $1,973,086.37 MXN
3. Tienda Anáhuac: $949,469.00 MXN
4. Tienda Lomas del Sur: $833,131.19 MXN
```
Fuentes: `ventas.csv` (agosto de 2026), `tiendas.csv` (tienda_id).

Qué decir: "Son ventas netas: ya descuentan las devoluciones, que en los datos vienen como importes negativos."

### A2 — Productos para reabastecer
**Pregunta:** `¿Qué productos de la tienda T01 están por debajo del punto de reorden?`
```
Inventario al 1 de septiembre de 2026 (máximo 30 resultados):
Tienda Valle Norte · SKU-1013 · Bolillo pieza: existencia 72, punto de reorden 260.
Tienda Valle Norte · SKU-1017 · Tortillas de maíz 1 kg: existencia 92, punto de reorden 150.
Tienda Valle Norte · SKU-1036 · Limón kg: existencia 59, punto de reorden 158.
```
Fuentes: `inventario.csv` (corte 2026-09-01), `catalogo_productos.csv`, `tiendas.csv`.

Qué decir: "Solo hay un corte de inventario, el 1 de septiembre; por eso lo dice siempre. Si pregunto por el inventario de julio, se abstiene."

### A3 — Tickets críticos
**Pregunta:** `¿Cuántos tickets de prioridad Crítica siguen abiertos y de qué tiendas son?`
```
Tickets críticos pendientes, por tienda:
Total: 0 tickets.
```
Fuentes: `tickets_mesa_servicio.csv` (prioridad y estado al corte), `tiendas.csv`.

Qué decir: "Cero es una respuesta calculada, no una falta de datos."
- **Contexto verificado aparte** (no lo dice el sistema): en los CSV hay 8 tickets de prioridad Crítica, 7 cerrados y 1 resuelto.
- Para mostrar un resultado distinto de cero: `¿Cuántos tickets están abiertos?` → Tienda Anáhuac 4, Tienda Cumbres 8, Tienda Valle Norte 5; **total 17** (también verificado aparte).

### A4 — No inventa cifras
**Pregunta:** `¿Cuánto vamos a vender en diciembre?` (`insufficient_data`)
```
No puedo obtener una cifra confiable de los CSV para esa pregunta. Las ventas disponibles abarcan únicamente junio a agosto de 2026.
```

### P1 — Contradicción entre documentos
**Pregunta:** `¿Cuántos días tiene un cliente para devolver una licuadora?`
```
Respuesta según los documentos:
- Tipo de producto: Electrodomésticos y electrónicos menores | Plazo para devolución o cambio: 30 días naturales | Requisitos: Ticket de compra, empaque original completo con accesorios y manuales. [procedimiento_devoluciones.pdf, p. 2, sección: 3. Plazos de devolución por tipo de producto]
- Para electrodomésticos y electrónicos menores como licuadoras, ... el cliente tiene 15 días con ticket ... [faq_gerentes_de_tienda.pdf, p. 4, sección: P15. ...] (no prevalece según la fuente; ver contradicción)
- El Ejecutivo de Servicio al Cliente autoriza hasta 500 pesos; ... [faq_gerentes_de_tienda.pdf, p. 4, sección: P17. ...] (no prevalece según la fuente; ver contradicción)
⚠️ Contradicción entre documentos (días): faq_gerentes_de_tienda.pdf p. 4: 15; procedimiento_devoluciones.pdf p. 2: 30. Prevalece procedimiento_devoluciones.pdf. faq_gerentes_de_tienda.pdf indica: "... prevalece siempre la política o el procedimiento vigente al que se hace referencia." ...
```
Qué decir: "Los dos documentos dicen cosas distintas. El sistema muestra ambos y solo da prioridad al procedimiento porque la propia FAQ dice que el procedimiento prevalece; si no lo dijera, lo dejaría sin resolver."

> ⚠️ **Defecto conocido (h8hyf2rw9y-afk/HEB#12, abierto):** la tercera viñeta (FAQ P17, montos de autorización) no corresponde al caso. Entra solo por estar en la misma página que la contradicción. Hay una corrección probada pendiente de autorización. Si no se ha aplicado al grabar, dilo en una frase o usa **P1-alternativa**.

**P1-alternativa** (sin contradicción): `¿Cuál es el plazo para devolver abarrotes?` → `Tipo de producto: Abarrotes y no perecederos | Plazo para devolución o cambio: 30 días | ...` `[procedimiento_devoluciones.pdf, p. 2, ...]`

### P2 — Respuesta directa con todas las restricciones
**Pregunta:** `¿Cómo se hace la devolución si el cliente pagó con tarjeta y no trae la tarjeta original?`
```
Respuesta según los documentos:
- Si el cliente pagó con tarjeta y no la presenta al momento de la devolución, el reembolso se realiza en vale de tienda. No se realizan reembolsos en efectivo de compras pagadas con tarjeta bajo ninguna circunstancia, ni reembolsos a una tarjeta distinta de la original. [procedimiento_devoluciones.pdf, p. 4, sección: 6.1 Cliente sin la tarjeta original]
```

### P3 — Dato de una tabla del PDF (opcional)
**Pregunta:** `¿Cuál es la tolerancia de merma en panadería?`
```
- Categoría: Panadería | Tolerancia mensual (% de venta): 3.5 | Umbral de alerta (%): 4.5 | Frecuencia de conteo: Diaria. [politica_mermas_y_caducidad.pdf, p. 3, sección: 4. Tabla de tolerancias de merma por categoría]
```

### P4 — No inventa políticas
**Pregunta:** `¿Cuál es la tolerancia de merma para electrónica?` (`insufficient_data`)
```
No encontré información suficiente en los documentos de políticas disponibles para responder esta pregunta. ...
ℹ️ La tabla «4. Tabla de tolerancias de merma por categoría» (politica_mermas_y_caducidad.pdf, p. 3) no incluye «electrónica»; incluye: Lácteos, Panadería, Carnes, Abarrotes, Bebidas, Limpieza, Frutas y verduras, Cuidado personal.
```

## Preguntas que **no** conviene usar en el video
Fallan hoy de forma conocida (ver [`LIMITATIONS.md`](LIMITATIONS.md)):

| Pregunta | Qué pasa |
| --- | --- |
| ¿Cuántas devoluciones sin ticket puede hacer un mismo cliente en doce meses? | Se enruta a datos y responde con tickets de la mesa de servicio, en vez de la política |
| ¿Cuánto tiempo puede permanecer en tienda un producto mermado después de su registro? | Se enruta a datos y responde con ventas por producto |
| ¿Con qué frecuencia se cuenta el inventario de abarrotes? | Se enruta a datos y responde con el resumen de reorden |
| ¿Cuántas unidades hay en el inventario de la tienda T03? | Responde un resumen de reorden, no unidades |

## Si algo falla durante la grabación
- "Sin conexión" en la barra lateral: la API no está activa; revisar la terminal 1.
- `Índice de políticas no disponible`: falta el paso 1 de la preparación.
- Errores de codificación en la consola de Windows: ejecutar `$env:PYTHONUTF8 = "1"` en esa terminal y volver a lanzar.
