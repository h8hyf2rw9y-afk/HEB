# Requisitos y verificación

**El estado actual frente al brief está en [`COMPLIANCE.md`](COMPLIANCE.md)**, la matriz requisito → implementación → evidencia → estado. Este archivo resume los objetivos y se mantiene solo como índice.

Fuente: [`../candidatos/BRIEF.md`](../candidatos/BRIEF.md) y [`../candidatos/datos/diccionario_datos.md`](../candidatos/datos/diccionario_datos.md). No se inventan plazos, umbrales ni listas de preguntas que no estén en el brief.

## Objetivos funcionales (estado al 2026-11-12)
- [ ] Preguntas en español sobre ventas con datos reales (DuckDB). ⏳ h8hyf2rw9y-afk/HEB#11
- [ ] Inventario y reabasto con la fecha de corte correcta (2026-09-01). ⏳
- [ ] Análisis de tickets de la mesa de servicio. ⏳
- [x] Políticas con evidencia citada: documento, página y sección.
- [x] Detección de contradicciones y precedencia solo cuando la fuente la declara. Hay dos inconsistencias sin detectar, documentadas en `LIMITATIONS.md`.
- [ ] No inventar cifras: ✅ en políticas · ⏳ en datos.

## Objetivos técnicos
- [ ] Separación UI / API / herramientas / RAG / LLM: falta la capa LLM.
- [x] Consultas deterministas y de solo lectura (motor DuckDB en lista blanca).
- [x] Validación de parámetros con Pydantic.
- [ ] Tokens, latencia, llamadas y ruta por pregunta: los campos existen, pero no hay LLM que los llene.
- [x] Pruebas deterministas y conjunto de evaluación representativo (políticas).
- [x] Documentación de instalación, reproducibilidad y limitaciones (`README.md`, `LIMITATIONS.md`).

## Entregables del brief
- [ ] README que corre en < 10 min: listo y medido en Linux; ⏳ falta probarlo en Windows.
- [ ] `plantilla_decisiones.md`: borrador en `DECISIONES_BORRADOR.md`; lo completa la persona que entrega.
- [ ] Video de 3–5 min: guion en `DEMO_SCRIPT.md`; se graba cuando estén las analíticas.
- [ ] Envío: fecha del brief, 25 de septiembre de 2026; ⚠️ confirmar si sigue vigente.

## Datos de origen
Los CSV y PDF de `candidatos/` son sintéticos y vienen con el ejercicio. Los datos privados y los índices generados no se suben (`.gitignore`).
