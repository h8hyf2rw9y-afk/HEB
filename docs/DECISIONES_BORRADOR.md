# Borrador para `candidatos/plantilla_decisiones.md`

> **Borrador preparado por Claude Code a partir del repositorio (2026-11-12, `main` en `020c23c`).** No sustituye a la plantilla original: la persona que entrega lo revisa, lo reescribe con su voz y lo copia a `candidatos/plantilla_decisiones.md`. Todas las cifras están medidas o verificadas en el repositorio. La plantilla pide 3 a 5 líneas por sección.

Nombre: `[PENDIENTE: persona que entrega]`
Fecha: `[PENDIENTE: fecha de envío]`
Repositorio: `[PENDIENTE: liga a GitHub o zip]`
Proveedor y modelos usados: **ninguno generativo**, por decisión de diseño. Datos: DuckDB con SQL fijo y parametrizado. Políticas: recuperación BM25 local con respuestas extractivas citadas. Embeddings con Ollama (`bge-m3`) implementados como opción, pero no activados ni probados con un modelo real.

## 1. Qué construí y por qué en ese orden
Primero los contratos entre piezas (FastAPI, interfaz `rag-v1`), para que interfaz, datos y políticas avanzaran en paralelo. Después las políticas, porque era la parte que podía "inventar" si se hacía mal: citas de documento, página y sección verificables, detección de la contradicción real FAQ (15 días) frente al procedimiento (30 días) y abstención con motivo. Luego los datos: DuckDB con un catálogo acotado de preguntas de ventas, inventario y tickets, siempre con SQL fijo y la fuente CSV en la respuesta. Al final, la interfaz Streamlit, el enrutamiento entre ambas vías y la verificación en Windows y Linux.

## 2. Qué decidí no hacer
- **No usar un LLM generativo:** cifras de SQL y políticas citadas textualmente, verificables y con 0 tokens. A cambio, el lenguaje que entiende es acotado.
- **No generar SQL a partir del texto ni responder pronósticos:** fuera del catálogo o del periodo (junio–agosto de 2026; inventario al 1 de septiembre), se abstiene.
- **No resolver contradicciones por fecha** ni por "el más reciente": solo cuando la fuente declara cuál prevalece.
- **No desplegar en la nube ni permitir cargar archivos;** tampoco subir datos ni índices al repositorio.

## 3. Cómo sé que funciona
- **Políticas:** 49 preguntas independientes, escritas antes de los últimos cambios: 46 correctas, 0 abstenciones falsas y 100 % de citas verificadas palabra por palabra contra el PDF. El conjunto de desarrollo da 63/63.
- **Datos:** las cifras de las preguntas de la demo se recalculan en las pruebas con Python puro, sin DuckDB, y coinciden. Ejemplo: agosto de 2026, Tienda Valle Norte, $2,163,062.36 MXN.
- **Pruebas:** 193 pasan en GitHub Actions en Windows y Linux, y `scripts/verify_delivery.py` comprueba la instalación de punta a punta.
- **Fallos conocidos y documentados:**
  - 3 preguntas de política mal enrutadas a datos;
  - 3 preguntas independientes de política que fallan;
  - 2 inconsistencias entre documentos que no se detectan.

  Todo está en `docs/LIMITATIONS.md`.

## 4. Costo por interacción y cómo lo medí
**0 llamadas a LLM, 0 tokens y $0 por tokens en cada interacción**, porque no se usa un modelo generativo. Medido con las métricas de `/chat` en 8 preguntas típicas: las de políticas responden en ~3–5 ms y las de datos en ~30–60 ms, solo CPU local. El sistema ya registra tokens y llamadas por pregunta; si se añadiera un modelo, el costo se mediría con esas mismas métricas. `[DECIDIR: si se confirma con la reclutadora que una solución sin LLM satisface "el modelo que hayas elegido"]`. Detalle en `docs/TOKEN_COSTS.md`.

## 5. Qué haría con dos semanas más
- **Mejorar el enrutamiento datos/políticas,** probándolo contra las 112 preguntas de política. Hoy 3 se desvían.
- **Ampliar el catálogo de preguntas de datos,** por ejemplo unidades de inventario por tienda o comparaciones entre meses, con una batería de evaluación calculada de forma independiente.
- **Medir embeddings reales** (Ollama `bge-m3`) frente a BM25 en el conjunto independiente. Los fallos actuales son de vocabulario; podrían mejorar, pero hay que medirlo.
- **Usar un LLM local solo para redactar sobre la evidencia citada,** midiendo tokens por interacción, y validar con el área responsable las dos inconsistencias entre documentos.
