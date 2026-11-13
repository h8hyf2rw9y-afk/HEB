# Interface contracts — draft v0

These are design contracts, not implemented endpoints. Both agents must agree on changes via GitHub Issue before modifying shared interfaces.

## HTTP
`POST /chat` request:
```json
{"question":"¿Qué tienda vendió más en agosto?","conversation_id":null}
```

Illustrative response shape:
```json
{
  "answer": "Respuesta fundamentada",
  "evidence": [{"source_type":"csv","source":"ventas.csv","reference":"2026-08"}],
  "metrics": {"input_tokens":0,"output_tokens":0,"total_tokens":0,"llm_calls":0,"latency_ms":0},
  "status": "answered"
}
```

Metrics above are placeholders, not measured values. `status` may be `answered`, `needs_clarification`, `insufficient_data`, or `error`.

Additive fields (backward compatible; existing keys are unchanged):
- PDF evidence items also carry `page`, `chunk_id` and `passage` (the retrieved text, whitespace-normalised, at most about 600 characters) so each citation can be audited.
- `metrics` also carries `llm_model`, `generation_latency_ms`, `generated`, `fallback_reason`, `answer_mode`, `tool_calls` (agent mode: tool, query executed, status, `duration_ms`, and `retried_with_original_question` when the model's reformulation found nothing) and `timings` (`total_ms`, `llm_ms`, `llm_load_ms`, `llm_prompt_eval_ms`, `llm_generation_ms`, `tokens_per_s`, `tools_ms`, `other_ms`, plus one entry per Ollama call).
- `GET /health/ollama` probes Ollama live (`reachable`, `model_available`); `GET /health/llm` only reports configuration.
- `GET /sources` lists the CSV files (rows, columns, date range) and the indexed PDFs; it is read-only.

## Internal tool interfaces
- `query_sales(filters: SalesFilters) -> ToolResult`
- `query_inventory(filters: InventoryFilters) -> ToolResult`
- `query_tickets(filters: TicketFilters) -> ToolResult`
- `retrieve_policy(query: str, top_k: int) -> list[Evidence]`

`ToolResult` must include structured data, source identifier, time period/as-of date, and errors. `Evidence` must include document name, page number, passage, and optional section. Input schemas and field names must be derived from the actual data dictionary before implementation.

## Cross-agent change protocol
Propose schema changes in an Issue, explain impact and migration, get agreement, update this file, and add contract tests before implementation.
