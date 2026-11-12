# Requisitos y verificación

**El estado actual frente al brief está en [`COMPLIANCE.md`](COMPLIANCE.md)**, la matriz requisito → implementación → evidencia → estado. Este archivo resume los objetivos y se mantiene solo como índice.

Fuente: [`../candidatos/BRIEF.md`](../candidatos/BRIEF.md) y [`../candidatos/datos/diccionario_datos.md`](../candidatos/datos/diccionario_datos.md). No se inventan plazos, umbrales ni listas de preguntas que no estén en el brief.

## Objetivos funcionales (estado al 2026-11-12)
- [x] Preguntas en español sobre ventas con datos reales (DuckDB, h8hyf2rw9y-afk/HEB#11), dentro de un catálogo acotado.
- [x] Inventario y reabasto con la fecha de corte correcta (2026-09-01).
- [x] Conteos de tickets de la mesa de servicio por prioridad, estado y tienda. Sin análisis del texto libre.
- [x] Políticas con evidencia citada: documento, página y sección.
- [x] Detección de contradicciones y precedencia solo cuando la fuente la declara. Hay dos inconsistencias sin detectar, documentadas en `LIMITATIONS.md`.
- [x] No inventar cifras: SQL fijo y citas textuales. Puede mostrar cifras reales que no responden a una pregunta mal enrutada (`LIMITATIONS.md` §2).

## Objetivos técnicos
- [x] Separación UI / API / herramientas / RAG; sin capa LLM, por decisión (ADR-010).
- [x] Consultas deterministas y de solo lectura (motor DuckDB en lista blanca).
- [x] Validación de parámetros con Pydantic.
- [x] Tokens, latencia y llamadas por pregunta (0 tokens, sin LLM). La ruta no se expone explícitamente.
- [x] Pruebas deterministas y conjunto de evaluación representativo (políticas).
- [x] Documentación de instalación, reproducibilidad y limitaciones (`README.md`, `LIMITATIONS.md`).

## Entregables del brief
- [ ] README que corre en < 10 min: medido en Linux (≈ 1 min) y con instalación y pruebas en Windows (CI); falta cronometrar el arranque en un Windows real.
- [ ] `plantilla_decisiones.md`: borrador en `DECISIONES_BORRADOR.md`; lo completa la persona que entrega.
- [ ] Video de 3–5 min: guion con salidas reales en `DEMO_SCRIPT.md`; pendiente de grabar.
- [ ] Envío: fecha del brief, 25 de septiembre de 2026; ⚠️ confirmar si sigue vigente.

## Datos de origen
Los CSV y PDF de `candidatos/` son sintéticos y vienen con el ejercicio. Los datos privados y los índices generados no se suben (`.gitignore`).
