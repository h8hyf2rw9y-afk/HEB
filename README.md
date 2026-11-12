# Asistente H-E-B para gerentes de tienda

Asistente en español para una gerente que administra cuatro tiendas: responde preguntas sobre **políticas internas** con citas de documento, página y sección, y dice cuándo no tiene información. Las **consultas analíticas** (ventas, inventario, tickets) están en desarrollo. Todos los datos y documentos son **sintéticos** y se entregaron con el ejercicio (`candidatos/`).

| Funcionalidad | Estado |
| --- | --- |
| Chat en español (Streamlit) + API (FastAPI) | ✅ |
| Políticas: respuestas citadas, contradicciones y abstención con motivo (4 PDF) | ✅ |
| Ventas, inventario y tickets con DuckDB | ⏳ En desarrollo (h8hyf2rw9y-afk/HEB#11). Hoy `/chat` responde "aún no disponible". |
| Costo en tokens con el modelo elegido | ⏳ Aún no hay LLM integrado (ver `docs/TOKEN_COSTS.md`) |

Estado completo frente al brief: [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md).

## Requisitos
- **Python 3.10 o superior** (probado con 3.13). En Windows, la instalación de [python.org](https://www.python.org/downloads/) con la opción "Add python.exe to PATH" incluye el lanzador `py`.
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
2. Instalar dependencias (~1 min con buena conexión):
   ```powershell
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r app/rag/requirements.txt -r frontend/requirements.txt fastapi "uvicorn[standard]" duckdb
   ```
3. Construir el índice de políticas (~10 s; se guarda en `data/rag_index`, ignorado por Git):
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

`$env:PYTHONUTF8 = "1"` evita errores de codificación al imprimir acentos en consolas Windows antiguas. Se aplica a la terminal donde se ejecuta; repítelo si abres otra.

## Instalación y arranque — macOS / Linux
```bash
git clone https://github.com/h8hyf2rw9y-afk/HEB.git && cd HEB
python3 -m venv .venv
.venv/bin/python -m pip install -r app/rag/requirements.txt -r frontend/requirements.txt fastapi "uvicorn[standard]" duckdb
.venv/bin/python -m app.rag build --pdf-dir candidatos/politicas --index-dir data/rag_index --embeddings off
.venv/bin/python scripts/verify_delivery.py
.venv/bin/python -m uvicorn app.api.main:app --port 8000          # terminal 1
.venv/bin/python -m streamlit run frontend/app.py                 # terminal 2
```

## Tiempo de puesta en marcha (medido)
| Paso | Contenedor Linux limpio, Python 3.13 |
| --- | --- |
| Crear venv + actualizar pip | 4 s |
| Instalar dependencias | 44 s |
| Construir índice | 7–8 s |
| `verify_delivery.py` | 8 s |
| API y Streamlit responden | < 5 s cada uno |
| **Total** | **≈ 1–2 min** |

**No se ha medido en Windows.** El objetivo del brief es menos de 10 minutos; la descarga de dependencias es el paso que más depende de la conexión.

Versiones probadas: fastapi 0.162.5, uvicorn 0.54.0, duckdb 2.0.0, pydantic 2.14.3, pymupdf 2.1, streamlit 1.67.0, httpx 0.28.1. El `requirements.txt` raíz todavía está vacío; consolidarlo está pendiente en h8hyf2rw9y-afk/HEB#13.

## Preguntas de ejemplo
- **Políticas:**
  - ¿Cuántos días tiene un cliente para devolver una licuadora? (muestra una contradicción entre documentos)
  - ¿Cómo se hace la devolución si el cliente pagó con tarjeta y no trae la tarjeta original?
  - ¿Cuál es la tolerancia de merma en panadería?
  - ¿Cuál es la tolerancia de merma para electrónica? (se abstiene e indica qué categorías existen)
- **Analíticas:** ⏳ en desarrollo. Ver el guion [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md).

## Pruebas
```bash
python -m pytest tests -q
```
Las pruebas de políticas usan los PDF de `candidatos/politicas`. La única prueba omitida necesita Ollama real (`HEB_RAG_OLLAMA_TESTS=1`). Evaluación de políticas por CLI:
```bash
python -m app.rag eval --cases tests/rag/fixtures/holdout_eval.json --index-dir data/rag_index --embeddings off --verify-pdf-dir candidatos/politicas
```

## Documentación de entrega
| Documento | Contenido |
| --- | --- |
| [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md) | Matriz requisito → implementación → evidencia → estado |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Arquitectura, RAG y BM25 |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Decisiones técnicas y su estado |
| [`docs/TOKEN_COSTS.md`](docs/TOKEN_COSTS.md) | Costo en tokens: medición actual y método |
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
