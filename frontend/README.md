# Frontend local (Streamlit)

## Iniciar

Desde la raíz del repositorio, con el entorno virtual activo:

```powershell
python -m pip install -r frontend/requirements.txt
python -m uvicorn app.api.main:app --reload
```

En una **segunda terminal**:

```powershell
python -m streamlit run frontend/app.py
```

Abre la URL local que indique Streamlit (normalmente `http://localhost:8501`).

La API predeterminada es `http://127.0.0.1:8000`. Puede modificarse en la barra lateral o con `HEB_API_URL`.

## Alcance de esta primera versión

- Chat en español, historial de sesión, estados de respuesta, fuentes y métricas.
- Indicador de salud de FastAPI.
- Sección de fuentes de datos que **describe** el flujo de importación futura; no hay subida de archivos funcional todavía.
- No almacena datasets ni PDFs, ni los sube a GitHub.
- El endpoint `/chat` aún devuelve `insufficient_data` hasta que se implemente la orquestación.

## Próxima fase: ingestión local de archivos

Diseñar con backend y RAG los endpoints y validaciones antes de habilitar el selector de archivos: formatos XLSX/CSV/PDF/ZIP, tamaño máximo, revisión de contenido, control de periodo y versión, rechazo de rutas peligrosas dentro de ZIP, aislamiento de datos, indexación y rollback. No permitir cargar archivos confidenciales a servicios externos sin autorización.
