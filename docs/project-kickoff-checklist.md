# Project Kickoff Checklist

Updated: 2026-03-29

Use this checklist to lock the product requirements before implementation.

## 1. Search Target

- [ ] Define the corpus type: docs, tickets, code, products, papers, or mixed
- [ ] Define whether ranking is at document, chunk, snippet, or entity-card level
- [ ] Define whether the system is internal-only, customer-facing, or a backend RAG service

## 2. Source Policy

- [ ] Confirm whether external web research is allowed
- [ ] List approved external domains if the web is allowed
- [ ] Confirm whether internal data sources are vector stores, databases, APIs, or MCP servers
- [ ] Define source trust tiers and fallback behavior when high-trust sources return nothing

## 3. Query Policy

- [ ] Define which queries should trigger research
- [ ] Define which queries must stay on the fast path
- [ ] Confirm whether users can force research mode manually
- [ ] Confirm whether users can see why research was triggered

## 4. Performance Budgets

- [ ] Set p50 and p95 latency targets for fast-path queries
- [ ] Set latency target or async policy for research-path queries
- [ ] Set max cost per query
- [ ] Set max number of external calls per research job

## 5. Ranking Quality

- [ ] Define the primary success metric: `NDCG@10`, `MRR@10`, or task-specific conversion
- [ ] Confirm whether click logs or relevance labels already exist
- [ ] Build an initial labeled evaluation set
- [ ] Define how freshness-sensitive queries will be labeled and measured

## 6. Trust and Explainability

- [ ] Confirm whether result citations are required
- [ ] Define whether ranking explanations are internal-only or user-visible
- [ ] Define how stale or conflicting evidence should be handled
- [ ] Define policy for hallucinated or unsupported research summaries

## 7. Security and Multi-Tenancy

- [ ] Confirm whether tenant isolation is required
- [ ] Confirm whether different tenants have different approved sources
- [ ] Define retention policy for query logs and research artifacts
- [ ] Define whether prompt-injection defense is required for web and file inputs

## 8. MVP Technical Defaults

- [ ] Backend: `Python + FastAPI`
- [ ] Retrieval: `Qdrant` hybrid retrieval
- [ ] Reranker: hosted API first
- [ ] Research worker: gated `o4-mini-deep-research`
- [ ] Planner: standard reasoning model with structured outputs

## 9. Release Readiness

- [ ] Offline baseline established
- [ ] Research-path win rate measured against baseline
- [ ] Cost and latency dashboards in place
- [ ] Safe fallback to fast path implemented
- [ ] Regression suite covers easy, hard, and freshness-sensitive queries
