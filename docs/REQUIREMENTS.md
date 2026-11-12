# Requirements and verification

**Status: provisional.** This document summarizes the project goals discussed with the user. The agents MUST inspect the original assignment brief, FAQ, and data dictionary before marking any requirement as confirmed. Do not invent deadlines, acceptance thresholds, or exact question lists.

## Functional goals
- [ ] Answer Spanish-language questions about store sales using actual structured data.
- [ ] Answer inventory and replenishment questions with the correct inventory snapshot date.
- [ ] Analyze service tickets using verified records.
- [ ] Answer questions about operational policies using cited PDF evidence (document and page).
- [ ] Identify conflicting policy passages, apply documented precedence when justified, and disclose uncertainty.
- [ ] Refuse to invent missing figures, policy terms, or current data when only historical snapshots exist.

## Technical goals
- [ ] Separate UI, API, query tools, document retrieval, and LLM orchestration.
- [ ] Use validated, read-only deterministic queries for calculations.
- [ ] Validate LLM tool-call parameters.
- [ ] Track input/output tokens, latency, number of model calls, and tool paths for each question.
- [ ] Include deterministic regression tests and a representative question evaluation set.
- [ ] Document setup, reproducibility, limitations, and model/runtime requirements.

## Deliverables to verify against original brief
- [ ] Exact required questions and expected response format.
- [ ] Deployment/offline execution requirements and permitted model/provider.
- [ ] Token-reporting format and any required cost calculations.
- [ ] Video, written report, repository and test requirements.
- [ ] Applicable submission deadline and data-sharing restrictions.

## Source data
Original CSV/PDF files should remain local or in approved private storage until sharing rights are confirmed. Do not add raw data to GitHub automatically.
