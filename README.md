# H-E-B Retail Assistant

Asistente conversacional en español para analizar datos de tiendas y consultar políticas operativas. **Ejercicio técnico de candidatura para H-E-B**: prototipo local, no una aplicación corporativa de producción.

**Tecnologías:** Python 3.11 · Streamlit · FastAPI · DuckDB + CSV · RAG sobre PDF · Ollama (LLM local).

## Inicio rápido: objetivo de 10 minutos

La meta es abrir el asistente en **menos de 10 minutos cuando ya están instalados Git, Python 3.11, Ollama y el modelo local**. La primera instalación de dependencias y la descarga del modelo (varios GB) pueden superar ese tiempo según el equipo y la conexión. **No se necesita API de pago.**

### Requisitos previos

| Programa | Verificar | Instalación |
| --- | --- | --- |
| Git | git --version | https://git-scm.com/downloads |
| Python 3.11 | Windows: py -3.11 --version / macOS-Linux: python3.11 --version | https://www.python.org/downloads/ |
| Ollama | ollama --version | https://ollama.com/download |
| Acceso al repositorio | El enlace de GitHub debe permitir clonar | Solicitar acceso si es privado |

Para la demostración recomendamos el modelo **qwen3:4b-instruct**. También se puede configurar otro compatible con Ollama; consulta la variable HEB_LLM_MODEL. Se necesita RAM y espacio para el modelo. CPU/GPU y descargas determinan la velocidad real.

**Datos incluidos:** candidatos/datos/ (CSV de ejemplo) y candidatos/politicas/ (PDF). Trabaja desde la **raíz del repositorio** para que el indexador encuentre los archivos.

## Windows 10/11 — PowerShell

**1. Obtener el proyecto y crear un entorno Python (terminal 1).**

    git clone https://github.com/h8hyf2rw9y-afk/HEB.git
    cd HEB
    py -3.11 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Si ya lo tenías clonado, desde HEB ejecuta:

    git fetch origin
    git switch main
    git pull --ff-only origin main
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Si falta el entorno .venv, créalo con py -3.11 -m venv .venv antes de instalar dependencias. No necesitas activar el entorno: todos los comandos indican su ejecutable Python.

**2. Comprobar Ollama y descargar el modelo si es necesario (terminal 1).**

    Invoke-RestMethod http://127.0.0.1:11434/api/tags
    ollama list
    ollama pull qwen3:4b-instruct

Si la primera comprobación falla, abre la aplicación Ollama desde el menú Inicio. Si no está funcionando en segundo plano, ejecuta **ollama serve** en **otra terminal** y mantenla abierta. No inicies dos servicios en el mismo puerto 11434. El comando pull solo requiere una descarga completa la primera vez.

**3. Construir el índice RAG una sola vez (terminal 1).**

    .\.venv\Scripts\python.exe -m app.rag build --pdf-dir candidatos/politicas --index-dir data/rag_index --embeddings off

Esta guía arranca en modo **BM25 sin embeddings** para evitar descargar un segundo modelo y reducir la configuración inicial. Si cambias los PDF, reconstruye el índice. El modo híbrido se explica en la documentación avanzada.

**4. Iniciar FastAPI (terminal 2, abierta en HEB).**

    $env:HEB_LLM_ENABLED = "true"
    $env:HEB_LLM_MODE = "agente"
    $env:HEB_LLM_MODEL = "qwen3:4b-instruct"
    $env:HEB_FAST_ANALYTICS = "true"
    $env:HEB_RAG_EMBEDDINGS = "off"
    $env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
    $env:HEB_LLM_TIMEOUT_S = "120"
    .\.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000

Deja esta terminal abierta. El backend queda en http://127.0.0.1:8000.

**5. Verificar los servicios (terminal 3).**

    Invoke-RestMethod http://127.0.0.1:8000/health
    Invoke-RestMethod http://127.0.0.1:8000/health/llm
    Invoke-RestMethod http://127.0.0.1:8000/health/rag
    Invoke-RestMethod http://127.0.0.1:8000/health/ollama

Debes obtener **status: ok**, **enabled: true**, **model: qwen3:4b-instruct** y **ready: true** en RAG. El endpoint /health/llm solo muestra la configuración del backend. **/health/ollama** sí consulta a Ollama en vivo: debe responder **reachable: true** y **model_available: true**.

**6. Iniciar Streamlit (terminal 4, abierta en HEB).**

    $env:HEB_CHAT_TIMEOUT_S = "540"
    .\.venv\Scripts\python.exe -m streamlit run frontend/app.py

Abre **http://localhost:8501**. La interfaz debe indicar que la API está conectada. Mantén abiertos los terminales de FastAPI y Streamlit mientras uses la aplicación.

## macOS / Linux — Terminal

Instala primero Git, **Python 3.11** y Ollama desde sus páginas oficiales. No dupliques el servidor Ollama si la aplicación ya lo está ejecutando.

**1. Clonar, instalar y construir el índice:**

    git clone https://github.com/h8hyf2rw9y-afk/HEB.git
    cd HEB
    python3.11 -m venv .venv
    ./.venv/bin/python -m pip install -r requirements.txt
    ollama list
    ollama pull qwen3:4b-instruct
    ./.venv/bin/python -m app.rag build --pdf-dir candidatos/politicas --index-dir data/rag_index --embeddings off

Si Ollama no se ejecuta como servicio, utiliza **ollama serve** en otra terminal.

**2. Iniciar API, en una nueva terminal situada en HEB:**

    export HEB_LLM_ENABLED=true
    export HEB_LLM_MODE=agente
    export HEB_LLM_MODEL=qwen3:4b-instruct
    export HEB_FAST_ANALYTICS=true
    export HEB_RAG_EMBEDDINGS=off
    export OLLAMA_BASE_URL=http://127.0.0.1:11434
    export HEB_LLM_TIMEOUT_S=120
    ./.venv/bin/python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000

**3. Iniciar Streamlit en una tercera terminal situada en HEB:**

    export HEB_CHAT_TIMEOUT_S=540
    ./.venv/bin/python -m streamlit run frontend/app.py

Abre http://localhost:8501. Para verificar salud, usa curl con las rutas /health, /health/llm, /health/rag y /health/ollama en http://127.0.0.1:8000.

## Prueba de funcionamiento (2–3 minutos)

Escribe estas preguntas en el chat. **Las preguntas de seguimiento deben hacerse en la misma conversación:**

1. ¿Qué tienda vendió más en agosto de 2026?
2. ¿Y cómo le fue en julio?
3. Compara sus productos con los de Cumbres.
4. ¿Cuál es el procedimiento para devolver un producto?
5. ¿Y si el cliente no tiene ticket de compra?

En **Fuentes** aparece cada cita con el pasaje exacto del PDF. En **Métricas de la consulta**, revisa el modelo, las llamadas LLM, si la respuesta fue generada, las consultas a herramientas, cualquier fallback y el tiempo por etapa (carga del modelo, lectura del prompt, generación, tokens/s y herramientas). La pantalla **Fuentes de datos** muestra los CSV, los PDF indexados y si Ollama responde. Las cifras vienen de los CSV y las políticas de los PDF, no de la memoria del modelo. **Las ventas solo abarcan junio–agosto de 2026; el inventario es un corte del 1 de septiembre de 2026.** La app debe reconocer preguntas que excedan las fuentes disponibles.

Si la respuesta indica **generated=false**, el sistema puede haber devuelto los datos verificados directamente porque Ollama falló, estaba desactivado o no superó la validación. No se deben presentar cifras inventadas como hechos.

## Probar el código (opcional)

Desde la raíz HEB:

**Windows:**

    .\.venv\Scripts\python.exe -m pytest tests -q

**macOS/Linux:**

    ./.venv/bin/python -m pytest tests -q

CI ejecuta pruebas en Windows y Ubuntu usando un **Ollama simulado**. Estas pruebas no certifican velocidad ni calidad del modelo real en otra computadora. Para evaluar Ollama local, con la API y Ollama activos:

**Windows:**

    .\.venv\Scripts\python.exe scripts\eval_agent_conversations.py --out data\eval_agent_conversations.json
    .\.venv\Scripts\python.exe scripts\eval_agent_conversations.py --episode siete_preguntas_reales --out data\eval_siete_preguntas.json
    .\.venv\Scripts\python.exe scripts\eval_llm_modes.py --out data\eval_llm_modes.json

**macOS/Linux:**

    ./.venv/bin/python scripts/eval_agent_conversations.py --out data/eval_agent_conversations.json
    ./.venv/bin/python scripts/eval_agent_conversations.py --episode siete_preguntas_reales --out data/eval_siete_preguntas.json
    ./.venv/bin/python scripts/eval_llm_modes.py --out data/eval_llm_modes.json

Para estimar cuánto costarían esos tokens con la API de Anthropic (simulación: el costo real por token es $0 porque el modelo corre localmente):

**Windows:**

    .\.venv\Scripts\python.exe scripts\simulate_api_cost.py data\eval_llm_modes.json

**macOS/Linux:**

    ./.venv/bin/python scripts/simulate_api_cost.py data/eval_llm_modes.json

Verifica manualmente la precisión y las citas PDF de las respuestas reales.

## Solución de problemas

| Problema | Solución |
| --- | --- |
| No se reconoce Python 3.11 | Instala Python 3.11 y vuelve a abrir la terminal. En Windows verifica py -0p. |
| No existe .venv o falta Streamlit | Crea el entorno e instala requirements.txt; usa siempre el Python de .venv. |
| Ollama no responde en 11434 | Abre Ollama o ejecuta ollama serve si no está activo. |
| Puerto 11434 ocupado al iniciar ollama serve | Probablemente ya hay un Ollama funcionando: consulta /api/tags antes de abrir otro. |
| El modelo no existe | Revisa ollama list; descarga qwen3:4b-instruct y comprueba HEB_LLM_MODEL. |
| Error WinError 10048 o puerto 8000 ocupado | Detén el proceso que ya usa ese puerto. No termines indiscriminadamente otros Python. |
| Streamlit muestra ReadTimeout | Verifica FastAPI y Ollama. El primer uso puede tardar; HEB_CHAT_TIMEOUT_S define la espera de la interfaz. |
| RAG no está ready | Repite el comando build del índice desde HEB y verifica que existan los PDF. |
| generated=false | Consulta fallback_reason y revisa la terminal de FastAPI; la respuesta de respaldo puede seguir siendo verificable. |
| Sin acceso al repositorio | Solicita acceso al propietario; un repositorio privado no se puede clonar sin permisos. |

**Detener:** Ctrl+C en las terminales de FastAPI y Streamlit. Ollama puede permanecer ejecutándose en segundo plano. Si cambias las variables de entorno, **reinicia FastAPI** para aplicar la nueva configuración.

## Cómo funciona

    Navegador -> Streamlit (8501) -> FastAPI (8000)
                                         |-- DuckDB -> CSV de tiendas, ventas e inventario
                                         |-- RAG -> PDF y referencias
                                         +-- Ollama (11434) -> comprensión y redacción
                                                |
                                      validación + fuentes + métricas

DuckDB usa consultas SQL parametrizadas o plantillas verificadas; el LLM no ejecuta SQL arbitrario. RAG recupera documentos de políticas con citas. Ollama redacta las respuestas cuando está disponible; si modifica cifras o políticas y no pasa validación, se muestra un respaldo verificable. El modo rápido HEB_FAST_ANALYTICS=true evita una llamada adicional al modelo para elegir herramientas en algunas preguntas exactas, **pero mantiene la redacción del modelo**.

### Variables principales

| Variable | Valor recomendado en esta guía | Función |
| --- | --- | --- |
| HEB_LLM_ENABLED | true | Activar generación local |
| HEB_LLM_MODEL | qwen3:4b-instruct | Modelo previamente descargado |
| HEB_LLM_MODE | agente | Selección de herramientas por el modelo cuando corresponde |
| HEB_FAST_ANALYTICS | true | Resolver consultas exactas directamente en DuckDB y después redactar |
| HEB_RAG_EMBEDDINGS | off | RAG BM25 sin modelo de embeddings |
| OLLAMA_BASE_URL | http://127.0.0.1:11434 | URL local de Ollama |
| HEB_LLM_TIMEOUT_S | 120 | Límite por llamada al modelo, en segundos |
| HEB_CHAT_TIMEOUT_S | 540 | Tiempo máximo de espera de Streamlit |
| HEB_LLM_VERIFIED_FIRST | true | Si DuckDB o el RAG ya responden con evidencia, el modelo solo redacta (una llamada); con false, el modelo elige primero la herramienta |
| HEB_LLM_KEEP_ALIVE | 30m | Tiempo que Ollama mantiene el modelo cargado entre preguntas |

### Alcance y limitaciones

Es una demostración de arquitectura y capacidades técnicas, **no un servicio autorizado para tomar decisiones reales en H-E-B**. Utiliza datos de ejemplo. No hay acceso a operaciones reales de tiendas, autenticación empresarial ni subida de nuevos archivos desde la interfaz. La calidad y latencia del modelo dependen del hardware y del estado del índice. Una respuesta con cita debe contrastarse con el documento oficial antes de usarse operativamente.

### Documentación adicional

- [Arquitectura](docs/ARCHITECTURE.md) y [contratos de API](docs/CONTRACTS.md)
- [Requisitos y verificación](docs/REQUIREMENTS.md)
- [Guía avanzada LLM/RAG](app/llm/README.md)
- [Decisiones](docs/DECISIONS.md) y [handoff](docs/HANDOFF.md)
- [Instrucciones de trabajo](AGENTS.md) y [Claude Code](CLAUDE.md)

**Seguridad:** no publiques credenciales, entornos virtuales ni archivos confidenciales del ejercicio. Los servicios se ejecutan en localhost y la inferencia se hace mediante Ollama local. El proyecto requiere revisión antes de fusionarse a main o desplegarse.
