# Asistente H-E-B para gerentes de tienda

Asistente en español para una gerente que administra cuatro tiendas. Responde con cifras calculadas desde los CSV sobre **ventas, inventario y tickets**, y con citas de documento, página y sección sobre **políticas internas**. Cuando no tiene información, lo dice. Todos los datos y documentos son **sintéticos** y se entregaron con el ejercicio (`candidatos/`).

| Funcionalidad | Estado |
| --- | --- |
| Chat en español (Streamlit) + API (FastAPI) | ✅ |
| Datos: ventas (jun–ago 2026), inventario (corte 2026-09-01) y tickets con DuckDB y SQL fijo parametrizado | ✅ Catálogo acotado de preguntas (ver «Alcance») |
| Políticas: 4 PDF con respuestas citadas, contradicciones y abstención con motivo | ✅ |
| Costo por interacción | ✅ 0 tokens: no se usa ningún LLM generativo, por decisión de diseño (ver [`docs/TOKEN_COSTS.md`](docs/TOKEN_COSTS.md)) |

Estado completo frente al brief: [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md).

## Requisitos
- **Python 3.11 o superior.** CI con 3.11 en Windows y Linux; probado también con 3.13. En Windows, la instalación de [python.org](https://www.python.org/downloads/) incluye el lanzador `py`.
- Git.
- No se necesita GPU, Ollama ni claves de API.

## Instalación y arranque — Windows (PowerShell)
Ejecuta cada bloque desde la carpeta del repositorio. Se usa `.\.venv\Scripts\python.exe` directamente, así que no hace falta activar el entorno ni cambiar la política de ejecución de PowerShell.

1. Clonar y crear el entorno:
   ```powershell
   git clone https://github.com/h8hyf2rw9y-afk/HEB.git
   cd HEB
   py -3 -m venv .venv
   ```
2. Instalar dependencias:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
3. Construir el índice de políticas (se guarda en `data/rag_index`, ignorado por Git):
   ```powershell
   $env:PYTHONUTF8 = "1"
   .\.venv\Scripts\python.exe -m app.rag build --pdf-dir candidatos/politicas --index-dir data/rag_index --embeddings off
   ```
4. Verificar (debe terminar con `fallos: 0`):
   ```powershell
   .\.venv\Scripts\python.exe scripts\verify_delivery.py
   ```
5. **Terminal 1**, la API:
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.api.main:app --port 8000
   ```
6. **Terminal 2**, la interfaz (abre `http://localhost:8501`):
   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run frontend/app.py
   ```

`$env:PYTHONUTF8 = "1"` evita errores de codificación con acentos en consolas Windows antiguas. Se aplica a la terminal actual; repítelo en la terminal 2 si lo necesitas.

## Instalación y arranque — macOS / Linux
```bash
git clone https://github.com/h8hyf2rw9y-afk/HEB.git && cd HEB
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m app.rag build --pdf-dir candidatos/politicas --index-dir data/rag_index --embeddings off
.venv/bin/python scripts/verify_delivery.py
.venv/bin/python -m uvicorn app.api.main:app --port 8000          # terminal 1
.venv/bin/python -m streamlit run frontend/app.py                 # terminal 2
```

## Tiempo de puesta en marcha (medido)
| Paso | Linux, entorno limpio (Python 3.13) | Windows, GitHub Actions (Python 3.11.9) |
| --- | --- | --- |
| Crear venv + actualizar pip | 4 s | — (el runner no crea venv) |
| `pip install -r requirements.txt` | 24 s | 62 s |
| Construir índice de políticas | 2–8 s | no medido aparte |
| `verify_delivery.py` | 3–8 s | no ejecutado en CI |
| `pytest` (backend + RAG) | 16 s | 22 s |
| API y Streamlit responden `/health` | < 1 s | **no medido** |
| **Total hasta usar la interfaz** | **≈ 1 min** | instalación y pruebas ≈ 1.5 min; arranque de API y Streamlit no medido |

El objetivo del brief es menos de 10 minutos. Según estas mediciones queda holgado; el paso que más depende de la conexión es la descarga de dependencias. **Pendiente:** cronometrar el arranque completo en un equipo Windows real (pasos 1–6).

## Preguntas de ejemplo
- **Datos:**
  - ¿Qué tienda vendió más en agosto de 2026?
  - ¿Qué productos de la tienda T01 están por debajo del punto de reorden?
  - ¿Cuántos tickets de prioridad Crítica siguen abiertos y de qué tiendas son?
  - ¿Cuánto vendimos en julio de 2026?
- **Políticas:**
  - ¿Cuántos días tiene un cliente para devolver una licuadora? (muestra una contradicción entre documentos)
  - ¿Cómo se hace la devolución si el cliente pagó con tarjeta y no trae la tarjeta original?
  - ¿Cuál es la tolerancia de merma en panadería?
- **Abstención:**
  - ¿Cuánto vamos a vender en diciembre? (no hay datos)
  - ¿Cuál es la tolerancia de merma para electrónica? (la tabla no incluye esa categoría)

Para reproducir las tres preguntas de datos del video sin levantar la API:
```bash
python scripts/demo_analytics.py
```

## Alcance de las consultas de datos
`app/tools/analytics.py` responde un **catálogo acotado** de intenciones con SQL fijo, nunca generado. Cubre:

- **Ventas netas** (incluidas las devoluciones), unidades o descuentos, por mes (junio, julio o agosto de 2026) o del periodo completo. Admite total, por tienda o ranking por tienda, producto, categoría o canal.
- **Inventario** al corte del 2026-09-01: productos en o por debajo del punto de reorden, y agotados.
- **Tickets:** conteos por prioridad y estado, por tienda.

Fuera de ese catálogo se abstiene; por ejemplo, con precios, costos, márgenes, pronósticos, comparaciones entre varios meses o meses sin datos. Limitaciones conocidas del enrutamiento entre datos y políticas: [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Pruebas
```bash
python -m pytest tests -q
```
- `tests/backend`: API, DuckDB y comprobaciones independientes de las cifras de la demo.
- `tests/rag`: políticas, con dos baterías de evaluación.
- `tests/docs`: la documentación coincide con el sistema.

La única prueba de RAG omitida necesita Ollama real (`HEB_RAG_OLLAMA_TESTS=1`).

## Documentación de entrega
| Documento | Contenido |
| --- | --- |
| [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md) | Matriz requisito → implementación → evidencia → estado |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Arquitectura, enrutamiento, DuckDB, RAG y BM25 |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Decisiones técnicas y su estado |
| [`docs/TOKEN_COSTS.md`](docs/TOKEN_COSTS.md) | Costo por interacción y decisión de no usar LLM |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | Fallos conocidos, contradicciones, Ollama sin probar |
| [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) | Guion del video de 3–5 minutos |
| [`docs/DECISIONES_BORRADOR.md`](docs/DECISIONES_BORRADOR.md) | Borrador para `candidatos/plantilla_decisiones.md` |
| [`app/rag/EVALUATION.md`](app/rag/EVALUATION.md) | Evaluación detallada del RAG |
| [`frontend/README.md`](frontend/README.md) | Detalles de la interfaz |

## Datos y privacidad
- Los datos y PDF de `candidatos/` son ficticios y vienen con el ejercicio.
- **No subir:** `.venv`, credenciales, índices (`data/rag_index`) ni datos privados. `.gitignore` ya los excluye.

## Desarrollo con dos agentes
Codex (datos y backend) y Claude Code (RAG) se coordinan por Issues y Pull Requests; el usuario aprueba los merges. Ver [`AGENTS.md`](AGENTS.md), [`CLAUDE.md`](CLAUDE.md), [`docs/HANDOFF.md`](docs/HANDOFF.md) y [`docs/CONTRACTS.md`](docs/CONTRACTS.md).
