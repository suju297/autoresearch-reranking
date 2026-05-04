# Autoresearch for reranking

## Executive summary

**Fact:** entity["people","Andrej Karpathy","ml engineer"]’s *autoresearch* repo demonstrates a very specific and unusually reproducible “autonomous experimentation” pattern: **one bounded editable surface**, **frozen data + evaluation harness**, **fixed per-run time budget**, **one primary metric**, and a **keep/discard/reset loop** recorded in a structured log. In the canonical repo, the agent edits only `train.py`, runs a 5‑minute experiment, parses a scalar metric (`val_bpb`), and either keeps the commit or resets it; the run is recorded in an untracked `results.tsv` following a prescribed schema. citeturn1view2turn1view0turn1view1

**Best definition (recommended):** *Autoresearch for reranking* is a **git-native**, **time-budgeted**, **autonomous hill-climbing + logging system** that improves a **multi-stage retrieval → reranking pipeline** by repeatedly proposing controlled changes *only within a small, explicitly allowed “reranking strategy surface”*—while a frozen evaluation harness measures ranking effectiveness (e.g., **NDCG@10**) and guardrails (latency/cost/memory). The human mostly edits the equivalent of `program.md` (the protocol), reviews outcomes, and decides when to merge the best “keep” commit.

**Closest existing “pieces”:**
- **Reranker models / APIs exist** (cross-encoders, BGE rerankers; hosted rerank endpoints). citeturn3search0turn6search1turn6search8turn5search0turn5search1turn5search2turn15search2  
- **Evaluation libraries exist** (TREC-style metrics via `pytrec_eval`, `ir_measures`, `ranx`; dataset access via `ir_datasets`; benchmark suites like BEIR, TREC DL). citeturn4search0turn4search1turn4search2turn4search3turn3search11turn7search23turn7search15  
- **LLM-based reranking toolkits exist** (notably RankLLM) and **LM-pipeline optimizers exist** (DSPy). citeturn9search1turn9search9turn10search1turn10search4  

**What’s missing (the differentiation):** a **small, controlled repo template** that makes **reranking improvements trustworthy under an autonomous agent** by:
1) freezing candidate generation + qrels + metric computation,  
2) enforcing time/cost budgets,  
3) using git + an experiment registry for provenance, and  
4) defining explicit keep/discard/rollback rules that optimize one metric while respecting guardrails.

**Best MVP path (4–6 weeks):** build a minimal repo where:
- `prepare.py` downloads a benchmark dataset (e.g., a BEIR slice) and **precomputes candidate pools** (BM25 and/or dense) and caches them,  
- the agent can only edit `rerank_strategy.py` (one file),  
- `eval.py` is frozen and computes **NDCG@10** as the primary metric (consistent with common IR practice and TREC DL guidance), with guardrails for latency/cost/memory, citeturn7search23turn7search3  
- the runner prints a fixed “grep-able” summary and records each attempt in `results.tsv` mirroring autoresearch. citeturn1view0turn1view2  

**Honest verdict:** it *can* be worth building if you focus the system on the hard part teams repeatedly struggle with—**rapid, reliable, auditable iteration on reranking choices** (model choice, top‑k, fusion, input formatting, thresholds) under real budgets. If you let the scope sprawl into “agentic web research” or end-to-end RAG answer quality too early, results will get noisy and the loop will stop being trustworthy. citeturn1view0turn7search23turn9search1

## Best definition of autoresearch for reranking

### Strong definition

**Fact:** In Karpathy’s repo, “autoresearch” is explicitly: a small repo where the agent modifies one file, runs a fixed-time experiment, evaluates with a frozen function, and keeps/discards changes as git commits, logging outcomes to `results.tsv`. citeturn1view2turn1view0  

**Proposed definition (for reranking):**

> **Autoresearch for reranking** is a *bounded autonomous experimentation system* that iteratively improves a multi-stage retrieval pipeline’s **reranking stage** (and a few tightly-scoped upstream knobs) by:  
> - allowing an agent to change **only** a constrained “reranking strategy surface”,  
> - running a **frozen evaluation harness** under a fixed per-run budget,  
> - measuring **one primary ranking metric** and several guardrails,  
> - and **advancing** only when changes measurably improve the metric without violating constraints—recording every attempt in a structured experiment log.

### What it optimizes

**Fact:** Standard offline ranking metrics for retrieval/reranking include **NDCG@k, MRR, MAP, Recall@k, Precision@k**, computed from qrels and runs; toolchains like `ir_measures` and `ranx` exist to standardize computation. citeturn4search1turn4search2turn4search31  

**Recommendation:** For v1, optimize:
- **Primary metric:** **NDCG@10** on a frozen dev set. (It is widely used for graded relevance and is explicitly recommended/used as a primary metric in TREC Deep Learning contexts; it’s also generally a better “overall ranking” signal than MRR when labels are sparse.) citeturn7search23turn7search3turn7search31  
- **Guardrails:** p95 latency, cost per query (if using APIs), memory/VRAM peak, and Recall@100 (or another recall-oriented metric) to detect “cheating via truncation” or overly aggressive filtering. citeturn4search31turn7search23turn1view0  

### Who it’s for

**Inference:** The best initial user is a **search/RAG engineer** who already has (or can create) a modest labeled dataset (queries + relevant docs) and wants to systematically improve ranking without manually running dozens of brittle experiments. This aligns with the “program.md as experimental protocol” interface Karpathy emphasizes. citeturn1view0turn1view2  

### What exact problem it solves that current stacks don’t

**Fact:** There are mature reranking components (cross-encoder rerankers, listwise LLM rankers, LTR plugins, hosted rerank endpoints). citeturn3search0turn6search1turn9search2turn6search2turn6search3turn5search0turn15search2  

**Inference:** The missing gap is less about “having a reranker” and more about **having a disciplined, reproducible, low-friction loop** that:
- forces every change through the *same* evaluator (frozen),  
- prevents uncontrolled surface-area changes (one editable module),  
- enforces fixed budgets (time/cost/latency), and  
- produces an auditable paper-trail (git + TSV) suitable for production review.

This is *exactly* the workflow-level contribution of autoresearch in the original repo. citeturn1view0turn1view2  

### Best starting use cases

Below is a practical ranking of your candidate use cases by **ease of building a trustworthy MVP** and **likelihood the loop produces real improvements**.

| Use case | Why it fits autoresearch | Main blocker | MVP suitability |
|---|---|---|---|
| **Code/document retrieval** | Public benchmarks + qrels are readily available; reranking metrics are well-defined; easy to reproduce across teams. citeturn3search11turn4search3turn7search14 | Doesn’t perfectly match “internal docs” domain. | **Best for MVP** |
| **RAG over internal docs** | High value; reranking strongly affects context quality; guardrails can include citation/grounding. citeturn15search22turn9search1 | Labeling/eval is the hard part; generation metrics add noise. | Strong follow-on |
| **Enterprise search** | Often already two-stage; can exploit behavioral data and LTR. citeturn6search2turn6search3turn6search10 | Needs click logs / counterfactual evaluation; offline qrels are costly. | Medium |
| **Web research agents** | Reranking is central; can tune for “source usefulness.” | Evaluation is hard and non-stationary; web content changes. | Low for v1 |
| **Ecommerce/product search** | Clear business metrics; classic LTR is common. citeturn14search1turn6search2turn15search0 | Needs behavioral/judgment data and online experimentation. | Medium/low |

**Recommendation:** Start with **code/document retrieval** because you can ship an MVP with a frozen benchmark harness using established datasets/metrics (BEIR, TREC DL style eval tooling), and later adapt the same framework to internal-doc RAG by swapping `prepare.py`/dataset loaders. citeturn3search11turn4search3turn7search23turn1view0  

## Mapping the autoresearch repo pattern to reranking

### What must carry over from Karpathy’s pattern

**Fact:** The essential mechanics in the original repo are explicitly documented in `README.md` and `program.md`: small repo, only one editable file, frozen `prepare.py`, fixed 5‑minute budget, one metric, git as memory, untracked TSV log, keep/discard via `git reset`, and a dedicated `autoresearch/<tag>` branch. citeturn1view2turn1view0  

### Concrete mapping table

| Autoresearch concept (Karpathy) | Reranking equivalent (recommended v1) | What is frozen vs editable |
|---|---|---|
| `prepare.py` (fixed constants, data prep, evaluation) citeturn1view2turn1view1 | `prepare.py`: downloads benchmark dataset, builds/caches candidate pools, defines splits, defines the metric function (e.g., NDCG@10), and defines guardrail metric computation. | **Frozen** (do not let agent edit) |
| `train.py` (only agent-edited surface) citeturn1view0turn1view2 | `rerank_strategy.py`: implements `rerank(query, candidates, ctx) -> ranked_list`, plus a *small allowed parameter space* (top‑k, fusion weights, truncation rules, model choice among an allowlist). | **Only editable file** for the agent |
| `program.md` (human protocol/instructions for agent) citeturn1view0turn1view2 | `program.md`: defines objective, constraints, allowed edit surface, how to run a trial, how to parse metrics, keep/discard rules, and “don’t touch” rules. | **Human-edited** |
| Fixed 5-minute budget citeturn1view0turn1view2 | Fixed evaluation budget per trial (e.g., 5 minutes wall-clock) including reranking + metric computation; timeout kill at 2× budget. | Enforced by **frozen runner** |
| Primary metric = `val_bpb` citeturn1view2turn1view0 | Primary metric = **NDCG@10** (or MRR@10 if you choose MS MARCO-style), computed on the dev set. citeturn7search23turn3search11turn4search31 | Frozen metric computation |
| `keep` / `discard` / `crash` statuses in `results.tsv` citeturn1view0 | Same statuses; `crash` includes timeout, OOM, exceptions, or missing metric line. | Frozen status assignment rules |
| `results.tsv` untracked schema citeturn1view0 | `results.tsv` untracked schema: commit, primary_metric, latency_p95, cost, memory, status, description. | Frozen schema; agent only appends rows |
| Baseline first run citeturn1view0turn1view2 | Baseline run should execute the default reranking strategy with cached candidate pools and log it as `keep baseline`. | Frozen “first run must be baseline” policy |
| Branch-per-run workflow citeturn1view0turn1view2 | Same. Use `autoresearch/<tag>` branch, each trial is a commit; discard via `git reset --hard <last_keep_commit>`. | Frozen git workflow guidelines |

### Bounded editable surface recommendation

**Key v1 recommendation:** **one file** (`rerank_strategy.py`) is the best bounded surface.

**Why (inference):**
- One file keeps the agent’s action space legible, enabling fast diff review—mirroring the original repo’s intent (“train.py is the only file you edit”). citeturn1view0turn1view2  
- Reranking “innovation” often comes from small but meaningful glue decisions (candidate slicing, score fusion, truncation strategy, model selection, rubric/prompt changes) that fit naturally into one strategy module. citeturn3search0turn8search4turn9search1  

### What is dangerous to let the agent modify

**Fact:** Karpathy’s repo explicitly bans edits to `prepare.py` and the evaluation harness, and bans adding dependencies. citeturn1view0turn1view2  

**Reranking-specific danger zones (recommend frozen):**
- **Dataset splits and qrels** (obvious reward hacking / overfitting risk).  
- **Metric computation code** (classic “optimize the scoreboard” failure mode).  
- **Candidate cache generation** (changes can silently alter evaluation comparability).  
- **Dependency manifest / network permissions** (blows up reproducibility and cost control).  

**Practical implication:** in v1, you want “`prepare.py` produces an immutable *evaluation package* (queries, qrels, candidates), and the agent only reorders.” This is also consistent with how multi-stage reranking is typically framed in IR: retrieve a candidate set, then rerank it. citeturn3search20turn7search11turn7search23  

## Existing tools and prior art

### What already exists, mapped to your categories

**Fact:** There is no single widely-adopted project that exactly matches “Karpathy-style autoresearch loop specifically for reranking,” but many projects cover important subcomponents. citeturn1view2turn9search1turn10search1turn4search1turn6search2  

The table below is intentionally biased toward **official docs, papers, and active GitHub repos**, per your constraints.

| Item | Category | What it does | Relevance to your project | Strengths | Limitations | OSS vs commercial | Role |
|---|---|---|---|---|---|---|---|
| entity["company","Cohere","ai model provider"] Rerank API | Hosted reranking API | Reranks a list of texts for a query, returns ranked results + scores. citeturn5search0turn5search16 | Provides plug-in reranker for your pipeline experiments. | High quality baseline; simple API contract. citeturn5search0 | Cost + latency; reproducibility depends on API stability/versioning. | Commercial | Dependency option |
| entity["company","Voyage AI","embedding rerank api"] reranker endpoint | Hosted reranking API | Query + documents → reranking results; documented REST + SDK. citeturn5search1turn5search9 | Another hosted reranker choice for agent to select. | Strong “component” framing for RAG stacks. citeturn5search9 | Same hosted constraints; cost controls needed. | Commercial | Dependency option |
| entity["company","Jina AI","search foundation models"] reranker API | Hosted reranking API | Hosted reranker models marketed for agentic RAG; API-based reranking. citeturn5search2turn5search14 | Provides a multilingual/code-focused reranking component. | Explicit benchmarks + positioning for “agentic RAG.” citeturn5search2 | Hosted variability; pricing limits; needs guardrails. | Commercial (API) | Dependency option |
| entity["company","Pinecone","vector db company"] rerank endpoint | Hosted reranking API | Hosted rerank endpoint for reranking results; part of inference API. citeturn15search2turn15search10 | Useful if your MVP uses Pinecone already. | Easy integration for two-stage retrieval. citeturn15search22turn15search2 | Vendor coupling; cost. | Commercial | Dependency option |
| entity["company","mixedbread.ai","search ai startup"] rerank models | Hosted + OSS models | Open-source rerankers + managed API; integration docs exist in ecosystem. citeturn15search19turn15search11turn15search15 | Gives both local + hosted paths for the same family. | OSS option improves reproducibility. citeturn15search11turn15search19 | Still needs careful evaluation + budget safeguards. | Both | Dependency option |
| entity["company","Hugging Face","ml model hub"] model hub | Open-source distribution | Hosts many rerankers (cross-encoders, BGE rerankers, etc.). citeturn6search1turn6search0 | Enables allowlisted model selection in a small repo. | Widely used; model cards & versioning. citeturn6search1turn5search18 | Quality varies; offline inference cost. | Mixed | Dependency |
| entity["organization","Sentence Transformers","embedding library"] cross-encoders | Open-source rerankers | Cross-encoder models for reranking and an API to run them. citeturn6search17turn6search5turn6search1 | Strong local baseline reranker; easy integration. | Many pretrained MS MARCO cross-encoders + docs. citeturn6search5 | CPU can be slow; GPU dependency for fast loops. | OSS | Dependency |
| entity["organization","BAAI","ai research institute"] BGE rerankers | Open-source rerankers | BGE rerankers output relevance scores from query-doc pairs; designed for reranking. citeturn6search8turn6search0turn6search34 | Strong modern reranker family; good for multilingual. | Explicit “reranker vs embedding model” guidance. citeturn6search8turn6search34 | Can be heavier models; needs truncation policy. | OSS | Dependency |
| entity["organization","FlashRank","reranking library"] | Open-source reranking library | Lightweight reranking library supporting cross-encoder + listwise LLM rerankers. citeturn5search3turn5search7 | Great for MVP speed (esp. CPU) and quick experiments. | Designed as drop-in reranker layer. citeturn5search3 | Still needs evaluation harness + budgets. | OSS | Dependency |
| entity["organization","Answer.AI","ml research org"] `rerankers` | Open-source aggregator | One library wrapping many rerankers: local cross-encoders, RankGPT/RankLLM, Cohere/Voyage/Jina APIs, etc. citeturn5search15 | Lets your repo stay small by delegating integrations. | Broad coverage across local+hosted. citeturn5search15 | You still need your own frozen evaluator + control logic. | OSS | Dependency/inspiration |
| entity["organization","OpenSearch","search engine project"] LTR plugin | Search experimentation / LTR | Second-stage reranking using XGBoost/RankLib models in OpenSearch. citeturn6search2turn6search10 | Shows “production reranking via rescoring” patterns. | Explicit judgment list + feature logging workflow. citeturn6search10 | Heavier infra; not a small repo; online signals needed. | OSS | Inspiration |
| entity["company","Elastic","search company"] LTR docs/plugin | Search experimentation / LTR | LTR as second-stage reranker; explains rescoring pipeline. citeturn6search3turn6search15 | Clear production framing of 2-stage ranking. | Infra complexity; not autonomous experimentation. | Mixed | Inspiration |
| entity["organization","Metarank","open source ranking service"] | Ranking optimization system | Open-source ranking service emphasizing personalization and LTR over search results. citeturn15search0turn15search16 | Shows end-to-end scoring service ideas; can be “advanced option.” | Not designed as a Karpathy-style bounded loop; needs event streams. | OSS | Inspiration/substitute (later) |
| entity["organization","Vespa","search engine"] phased ranking | Multistage retrieval framework | Two-phase ranking with bounded rerank count; supports ML models in ranking expressions. citeturn15search1turn15search5turn15search25 | Very aligned to “expensive second phase with predictable bounds.” | Infra-heavy for MVP; not agentic by default. | OSS + hosted | Inspiration (advanced) |
| entity["organization","Pyserini","ir toolkit"] | Multistage retrieval framework | Reproducible IR toolkit supporting sparse+dense retrieval via Lucene + Faiss. citeturn12search0turn12search14turn12search6 | Great reproducibility; integrates common datasets and eval scripts. | Java/Lucene complexity; heavier setup. citeturn12search12 | OSS | Dependency (balanced/advanced) |
| entity["organization","PyTerrier","ir experimentation toolkit"] | Multistage retrieval + eval | Declarative pipelines and experiments; uses `ir_measures`; supports rerankers and dense retrieval extensions. citeturn12search1turn12search23turn12search15 | Very aligned with “pipeline + evaluation” mindset. | Still larger than “tiny repo” ideal. | OSS | Inspiration/dependency |
| entity["organization","DSPy","lm pipeline optimizer"] | LLM program optimizer | Compiles LM pipelines to maximize a metric; has optimizers that tune instructions/demos. citeturn10search1turn10search8turn10search4 | Close conceptual neighbor: “optimize pipeline to metric.” | Typically tunes prompts/demos, not arbitrary reranking code under a strict “one file” boundary. | OSS | Inspiration/substitute for some scopes |
| entity["organization","Optuna","hpo framework"] | Experimentation / HPO | Black-box hyperparameter optimization framework. citeturn10search2turn10search6 | Useful “engine” for parameter search (top‑k, weights). | Doesn’t give you the “agent edits code” workflow or git-native provenance. | OSS | Dependency (optional) |
| entity["company","Weights & Biases","mlops platform"] sweeps | Experiment tracking / tuning | Hyperparameter sweeps (bayes/grid/random) with tracking. citeturn10search3turn10search15 | Great if you outgrow TSV logging. | Heavy compared to autoresearch minimalism. | Commercial + free tier | Optional later |
| entity["company","OpenAI","ai company"] MLE-bench | Autonomous experimentation benchmark | Benchmarking for ML engineering agents; provides preparation & grading scripts. citeturn11search6turn11search2 | Strong prior art for “agents + grader scripts.” | Not reranking-specific. | OSS | Inspiration for scaffolding |
| entity["organization","SWE-agent","coding agent project"] | Autonomous code experimentation system | Agents use tools to navigate repos and run tests; paper emphasizes interface design. citeturn11search7turn11search0 | Good inspiration for agent-computer interface ergonomics. | Your project needs *metric-driven experimentation*, not issue fixing. | OSS | Inspiration |
| entity["organization","RankLLM","llm reranking toolkit"] | LLM reranking toolkit | Toolkit + paper for reproducible LLM-based reranking; includes evaluation and prompt analysis modules. citeturn9search1turn9search9turn9search13 | One of the closest building blocks for “LLM-as-reranker” experiments. | Not an autonomous keep/discard repo loop. | OSS | Dependency/inspiration |
| entity["organization","BEIR","ir benchmark suite"] | Retrieval benchmark suite | Benchmark + framework for evaluating retrieval models across tasks. citeturn3search11turn3search3 | Great “frozen eval harness” basis for MVP. | Full BEIR is large; need careful subset selection for fast loops. | OSS | Core benchmark |
| entity["organization","MIRACL","multilingual ir benchmark"] | Retrieval benchmark suite | Multilingual IR dataset with many queries and judgments. citeturn7search0turn7search8 | Good later for multilingual reranking. | Too heavy for initial tight budget MVP. | OSS | Follow-on benchmark |

### Is there already a true “autoresearch for reranking” system?

**Fact:** Karpathy’s repo is about autonomous experimentation on LLM training, not reranking. citeturn1view2turn1view0  
**Fact:** RankLLM focuses on LLM-based reranking + integrated evaluation and prompt analysis, but it is not structured as a “single editable surface + keep/discard/reset by git” loop. citeturn9search1turn9search13turn9search9  
**Fact:** DSPy compiles LM pipelines to maximize a metric, but its editable surface and optimization mechanisms differ materially from Karpathy’s “one target file changed by an agent under strict budgets.” citeturn10search1turn10search8  

**Inference (answer):** There is no canonical, widely adopted “autoresearch for reranking” repo template yet; the closest practical combination is: **RankLLM (reranking component)** + **BEIR/TREC-style eval tooling** + **an experiment controller** (Optuna/W&B or custom) + **your desired bounded-edit + git keep/discard protocol**.

## Search space for autonomous reranking experiments

### Where reranking systems actually have tunable leverage

**Fact:** Classic multi-stage IR is: retrieve candidate set (often BM25), then rerank with a stronger neural model; BERT-based reranking is a standard instance of this framing. citeturn3search20turn7search11turn7search23  
**Fact:** Reranking approaches span pointwise/cross-encoder (e.g., BERT reranking), sequence-to-sequence ranking (MonoT5), and listwise methods including LLM rankers (RankVicuna/RankZephyr) and ListT5. citeturn3search0turn3search1turn9search2turn9search0turn9search4  

### A practical “agent search space” matrix

The table below is not theoretical—it’s what you can realistically expose in a bounded strategy surface while still getting stable, interpretable experiments.

**Ratings are inference**, grounded in known cost/effectiveness tradeoffs of reranking, fusion, and evaluation constraints described in IR literature and tool documentation. citeturn3search20turn8search4turn9search1turn7search23  

| Lever the agent could change | Relevance upside | Runtime/cost risk | Stability / reproducibility | Ease of autonomous experimentation | MVP suitability | Notes / why it matters |
|---|---|---|---|---|---|---|
| Top‑k candidate count before reranking | Medium–High | Medium (reranker scales with k) | High if deterministic | High | **High** | The second-stage budget is often a dominant knob; bounded by time budget just like autoresearch. citeturn1view0turn3search20 |
| Hybrid retrieval weights (BM25 + dense) | Medium | Low–Medium | High if candidates cached | Medium | High (if precomputed) | Fusion like RRF avoids score normalization issues and is robust across systems. citeturn8search4turn8search7turn8search15 |
| BM25 + rerank | High baseline | Medium | High | High | High | Common two-stage baseline in IR and production. citeturn3search20turn6search15 |
| Dense retrieval + rerank | Medium–High | Medium | Medium (ANN nondeterminism) | Medium | Medium | Great later; but for MVP, prefer caching or deterministic indices. citeturn12search0turn12search13 |
| Score fusion (RRF vs weighted sum vs normalization) | Medium | Low | High | High | **High** | RRF is widely documented and implemented in major search stacks. citeturn8search4turn8search11turn8search15turn8search7 |
| Metadata-aware reranking (recency, doc type) | Medium | Low | Medium–High | Medium | Medium | High value in enterprise/internal docs; needs metadata in dataset/harness. citeturn6search15turn6search32 |
| Cross-encoder rerankers | High | Medium | High if fixed seeds | Medium | **High** | Strong improvements; canonical “reranking with BERT” style. citeturn3search0turn6search17 |
| LLM-as-reranker (listwise) | Medium–High | High (latency/cost) | Medium (API nondeterminism) | Medium | Low–Medium | RankLLM exists, but reproducibility/cost control is harder; needs strict guardrails. citeturn9search1turn9search13turn5search0 |
| Pairwise vs listwise ranking | Medium | Medium–High | Medium | Medium | Medium | Listwise adds positional bias + formatting concerns; ListT5 provides efficient frameworks but is heavier. citeturn9search4turn9search2turn3search13 |
| Cascade thresholds (early exit) | Medium | Low–Medium | High | Medium | Medium | Useful for latency budgets: rerank only when needed; mirrors “phased ranking” patterns. citeturn15search1turn15search25 |
| Query rewriting before retrieval/reranking | Medium–High | Medium (extra LLM calls) | Medium | Medium | Medium | Techniques like HyDE and multi-query/RAG-fusion can improve recall; but can add noise. citeturn13search1turn13search0turn8search4 |
| Candidate filtering & deduplication | Medium | Low | High | High | **High** | Easy, cheap wins: remove near-duplicates, enforce doc diversity; reduces wasted reranker budget (inference). |
| Prompt/rubric changes for LLM reranking | Medium | Medium–High | Medium | High | Medium | Works well when you already accept LLM reranking; requires stable prompt parsing. citeturn9search1turn5search7 |
| Feature engineering for classical LTR (LambdaMART) | Medium–High | Medium (data/logging) | High | Low–Medium | Low for MVP | Powerful in enterprise search with logs; requires substantial scaffolding. citeturn14search1turn6search2turn6search10 |
| Domain-specific reranker selection (routing) | Medium–High | Medium | Medium | Medium | Medium | E.g., pick code reranker vs general; needs label stratification / metadata. citeturn5search2turn7search0turn6search6 |

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["two-stage retrieval reranking pipeline diagram bm25 cross-encoder","reciprocal rank fusion hybrid search diagram","LLM listwise reranking prompt diagram"]}

### What the agent should be allowed to change in v1

**Recommendation (tight v1 scope):** allow only changes that satisfy all three:
1) can be expressed inside one strategy module,  
2) can be evaluated deterministically from frozen cached candidates + qrels,  
3) is likely to produce measurable improvements in <5 minutes.

That usually means:
- candidate_k (and optional per-source candidate_k if you have multiple retrieval sources),  
- fusion method + weights (RRF constant, per-source weights), citeturn8search4turn8search7turn8search15  
- reranker selection among an allowlist (e.g., one small cross-encoder + one BGE reranker), citeturn6search1turn6search8turn6search34  
- input formatting/truncation strategy (head-only vs head+tail, max tokens), because cross-encoders are length-sensitive (inference grounded in model context constraints), citeturn6search17turn5search24  
- optional cheap filtering/dedup rules.

### What should stay fixed until later

**Recommendation:** keep these fixed in v1 to avoid noisy/untrustworthy experiments:
- changing the retrieval backend/index construction itself (too many confounds),  
- changing dataset, qrels, or splitting logic,  
- adding online signals (click logs) or counterfactual training,  
- end-to-end RAG answer generation metrics (hard to stabilize early).

**Fact:** Even in IR research, reranking pipelines are recall-limited by their initial candidate pool; methods like GAR address this but introduce additional complexity. citeturn14search0turn14search6turn14search7  
**Inference:** For a small controlled repo, you want to *first* get stable improvements on a fixed candidate pool; adaptive retrieval can be an “advanced version.”

## Autonomous loop design

### A concrete experiment lifecycle

This is a direct adaptation of the system described in Karpathy’s `program.md`, but specialized to reranking, including cost/latency guardrails. citeturn1view0turn1view2  

**Frozen components (trust boundary):**
- `prepare.py`: creates a local cache directory containing:
  - frozen query/dev/test splits,
  - qrels,
  - cached candidate pools (e.g., BM25 top‑200 per query and optional dense top‑200),
  - frozen document texts (or stable doc IDs + a consistent `docstore` lookup),
  - evaluator code for NDCG@10 + guardrails. citeturn4search3turn4search1turn3search11turn7search23  
- `eval.py` (or evaluator functions inside `prepare.py`): computes metrics from the strategy’s output; immutable.

**Editable surface (agent):**
- `rerank_strategy.py`: a single file that exports something like:
  - `def rerank(query: str, candidates: list[Candidate], ctx: Context) -> list[Candidate]`
  - plus a small config dict / constants inside the file (candidate_k, fusion weights, chosen model id from allowlist).

**Per-trial run protocol (mirrors autoresearch):**
1) Inspect current branch and last “keep” commit.  
2) Make one small strategy change (or a tightly scoped new heuristic).  
3) `git commit -m "short description"`  
4) Run the evaluator with a **hard timeout** (e.g., 600s).  
5) Parse a fixed “summary block” from stdout.  
6) Append a row to `results.tsv` (untracked).  
7) **Keep**: if primary metric improved and guardrails pass.  
8) **Discard**: if worse or equal (or improvement too small / not significant). Reset branch to last keep commit.  
9) **Crash**: if run errors/timeouts/OOM; log crash values, reset.  
10) Repeat until budget exhausted.

**Fact:** This is exactly the loop described in Karpathy’s `program.md`, including kill conditions, crash handling, and `git reset` rollback. citeturn1view0  

### Keep/discard rules and why they must be explicit

**Fact:** Karpathy’s repo uses a simple primary metric threshold (improve `val_bpb`) plus qualitative criteria like VRAM soft constraint and “simplicity criterion.” citeturn1view0  

**Recommended reranking keep rule (v1):**
- Keep only if:
  - `Δ NDCG@10 >= +0.002` absolute on dev (tunable), **and**
  - p95 latency does not exceed `baseline_p95 * 1.10`, **and**
  - peak memory does not exceed `baseline_mem * 1.15`, **and**
  - Recall@100 does not drop more than 0.01 absolute (or another recall guardrail), **and**
  - (optional but recommended) improvement is stable under bootstrap resampling or a paired test (ranx supports statistical testing workflows; PyTerrier and ir_measures are commonly used in experimental workflows). citeturn4search2turn12search1turn4search1  

**Inference:** In reranking, “equal or slightly better” changes can be spurious due to query-set sampling effects or nondeterministic model inference (especially for LLM APIs). So your rule needs a *minimum effect size* and ideally *stability checks*.

### Run marked “crash” conditions

Use deterministic, machine-checkable rules:
- non-zero exit code,
- timeout exceeded,
- OOM detected,
- missing metric line in output (Karpathy uses “if grep output empty, run crashed”). citeturn1view0  

### Logging design

**Fact:** Karpathy specifies `results.tsv` with fixed columns (commit hash, metric, memory, status, description), and explicitly says the TSV is untracked and tab-separated. citeturn1view0  

**Recommended `results.tsv` schema (reranking):**

| Column | Type | Meaning |
|---|---|---|
| `commit` | str | short git hash |
| `ndcg@10` | float | primary metric (0.000000 on crash) |
| `recall@100` | float | guardrail recall |
| `p95_ms` | float | p95 end-to-end query latency on eval set |
| `cost_usd` | float | estimated evaluation cost (0 if local) |
| `peak_mem_gb` | float | RAM/VRAM peak |
| `status` | enum | keep/discard/crash |
| `description` | str | what changed |

Also write untracked artifacts per run:
- `runs/<commit>/run.log` (stdout/stderr),
- `runs/<commit>/metrics.json` (machine-readable metrics + config hash),
- `runs/<commit>/diff.patch` (optional) for easy review.

### Safeguards for trustworthiness

This is the part that matters most for reranking because “leaking labels” is easier than in LLM pretraining.

**Reward hacking the eval harness**
- **Minimal v1 safeguard (practical):** freeze evaluator code and keep qrels private to the evaluator function; do not pass qrels into strategy.  
- **Stronger safeguard (recommended if you run truly untrusted agents):** execute `rerank_strategy.py` inside a sandboxed subprocess/container with a restricted filesystem view where qrels are not readable (OS-level isolation).  
**Inference:** Python-level “monkeypatching open()” is not robust against a creative agent; OS-level sandboxing is the real line. (Karpathy’s repo relies on norm compliance rather than hardened isolation, but reranking is more vulnerable to label overfitting.) citeturn1view0turn7search23  

**Overfitting to dev**
- Keep a held-out test split and do not let the agent make keep/discard decisions on it. Only the human runs test evaluation for final selection.  
- Consider periodic “shadow test” checks (every N keeps) and revert if test regresses beyond tolerance.

**Instability across repeated runs**
- For deterministic cross-encoders, fix seeds and evaluation order.  
- For LLM rerankers (RankLLM / hosted APIs), run at `temperature=0` and repeat evaluation 2–3×, keeping only if mean improves and variance is bounded. RankLLM explicitly discusses reliability concerns and includes analysis modules for prompts/responses. citeturn9search1turn9search13turn9search17  

**Latency/cost blowups**
- Enforce per-trial wall-clock timeouts (as in autoresearch). citeturn1view0  
- Track p95 latency and a cost proxy (tokens × price, or API call count).  
- Hard stop if the experiment budget is exceeded.

**Pathological improvements that hurt maintainability**
- Add a “simplicity” preference similar to Karpathy’s (prefer simpler diffs when improvements are tiny). citeturn1view0  
- Enforce a max-diff rule: e.g., strategy file changes > 100 lines require human approval.

## Evaluation and metrics

### What metrics are best practice for reranking

**Fact:** Standard IR evaluation includes MAP, MRR, NDCG, precision/recall at cutoff; `ir_measures` and `pytrec_eval` are specifically built to avoid ad-hoc reimplementations. citeturn4search0turn4search1turn4search31  
**Fact:** `ranx` is designed for fast ranking evaluation and includes metric computation and statistical testing support (and also includes fusion functionality and “fusion optimization”). citeturn4search2turn4search10  

### Recommended v1 metric set

**Primary metric (v1):** **NDCG@10**  
- **Fact:** TREC Deep Learning Track guidance explicitly highlights NDCG@10 as the primary metric, and discusses metric disagreement artifacts (e.g., MRR interacting with sparse labels and “oldness”). citeturn7search23turn7search3turn7search31  

**Guardrail metrics (2–5):**
1) **Recall@100** (or NCG@100 / Recall@k): catches regressions where you improve top ranking but lose coverage. citeturn7search23turn4search31  
2) **p95 latency** (ms/query for rerank stage + total pipeline). (Practical production constraint; also enforces fairness under fixed budgets.) citeturn1view0turn15search1  
3) **Cost per query** (if using hosted rerankers): count calls and/or tokens. citeturn5search0turn15search2  
4) **Peak memory / VRAM**: prevents “win by blowing up compute.” Karpathy treats VRAM as a soft constraint; you can do the same. citeturn1view0  
5) (Optional) **MRR@10** as a secondary interpretability metric, but do not optimize it as primary on sparse-label datasets. citeturn7search3turn7search31  

### Dataset sizing and splits for fast, trustworthy loops

**Fact:** BEIR is broad and can be heavy; IR tooling like `ir_datasets` exists specifically to standardize dataset access (docs, queries, qrels). citeturn3search11turn4search3turn4search15  

**Recommendation (v1):**
- Use a **small-but-nontrivial dev set**: 200–1,000 queries depending on hardware; keep it fixed.  
- Keep a **test set** of similar size (or larger) that the agent never uses for keep/discard.  
- Start with **one dataset** to minimize confounds; add multi-dataset evaluation later to reduce overfitting.

**Inference:** If you try to optimize across many datasets in v1, you will either (a) exceed your time budget or (b) increase noise so much that keep/discard is uninformative.

### Detecting false improvements

Practical approaches:
- Bootstrap confidence intervals over per-query NDCG deltas (paired); require improvement beyond noise band.  
- Re-run “keeps” twice (same commit) to verify stability (especially if any nondeterminism).  
- Maintain a “shadow canary set” (small random subset) and monitor for regressions.

### Structured registry format

**Fact:** Karpathy’s `results.tsv` is a lightweight registry designed to be grep-friendly and untracked, with strict column definitions. citeturn1view0  

**Recommendation:** Keep the compact TSV for the autoresearch feel, and add `runs/<commit>/metrics.json` as the canonical machine-readable artifact so you can later build dashboards without changing the core loop.

## Proposed repo design

### Smallest clean repo shape

This is intentionally “Karpathy-like”: minimal, frozen evaluator, one editable surface, and a TSV registry.

**Recommended tree:**
- `README.md` (frozen)
- `program.md` (**human edited**, protocol)
- `prepare.py` (**frozen**)  
- `eval.py` (**frozen**)  
- `run.py` (**frozen**, one experiment runner, prints summary block)  
- `rerank_strategy.py` (**agent edited**, only file)
- `pyproject.toml` (frozen deps; use `uv` if you want parity with autoresearch) citeturn1view2turn1view0  
- `results.tsv` (untracked)
- `runs/` (untracked)

**File responsibilities (who edits what):**
- `program.md`: the only place the human “programs the researcher.” It should include:
  - objective: “maximize NDCG@10”
  - constraints: “edit only rerank_strategy.py; don’t change eval”
  - the run command, how to parse `ndcg@10` from stdout
  - keep/discard rule and guardrails
  - stop conditions / budget
  This mirrors how Karpathy frames `program.md` as the real interface. citeturn1view0turn1view2  
- `prepare.py`: downloads dataset and prepares cached evaluation artifacts (candidate pools, docstore). Keep it immutable like Karpathy’s `prepare.py`. citeturn1view2turn1view1  
- `eval.py`: computes NDCG@10 + guardrails using `ranx` or `ir_measures`. citeturn4search1turn4search2  
- `rerank_strategy.py`: contains the *only* editable surface.

### Naming conventions

- Branch: `autoresearch/<tag>` (date-based tag) – matches the original pattern. citeturn1view0  
- Run artifacts: `runs/<short_commit>/run.log`, `runs/<short_commit>/metrics.json`.

## Architecture options

### Very small MVP

- **Corpus/input:** a small BEIR slice or similar dataset with qrels. citeturn3search11turn4search3  
- **Retrieval layer:** frozen precomputed candidate pools (BM25 top‑K) to keep runtime bounded (inference: easiest way to stay tiny).  
- **Reranker layer:** local cross-encoder (small) or FlashRank CPU reranker. citeturn6search1turn5search3  
- **Orchestration:** one Python runner with timeout.  
- **Evaluation:** `ranx` or `ir_measures`. citeturn4search2turn4search1  
- **Storage/logging:** `results.tsv` + run logs.  
- **Tradeoff:** fastest path to “true autoresearch feel,” but less end-to-end realism than a full retrieval backend.

### Balanced practical version

- **Corpus:** BEIR dataset(s) selected for moderate size. citeturn3search11  
- **Retrieval:** BM25 + dense retrieval (either deterministic or cached). Use RRF fusion. citeturn8search4turn8search15turn8search11  
- **Reranker:** allowlist of 2–4 rerankers (cross-encoder + BGE reranker). citeturn6search1turn6search8  
- **Evaluation:** NDCG@10 primary + recall/latency guardrails. citeturn7search23  
- **Tradeoff:** more realistic; slightly heavier setup.

### Advanced version

- **Retrieval backend:** OpenSearch/Elasticsearch/Vespa with real indexing and rescoring; optionally LTR models (LambdaMART) and online signals. citeturn6search2turn6search3turn15search1turn14search1  
- **Reranker:** LLM listwise rerankers via RankLLM + prompt analysis modules. citeturn9search1turn9search13  
- **Evaluation:** offline + online experimentation, counterfactual/IPS if you go there (beyond MVP).  
- **Tradeoff:** becomes a real search relevance platform; much larger than a tiny controlled repo.

## Build vs buy and MVP recommendation

### Build vs buy (practical)

**Fact:** There are plentiful hosted rerank APIs and open-source reranker models; there are also robust evaluation packages and dataset toolchains. citeturn5search0turn5search1turn6search1turn4search1turn4search3turn3search11  

**Recommendation:**  
- **Build from scratch (core differentiation):**
  - the **bounded-edit repo template**,
  - the **frozen eval harness + caching**,
  - the **experiment runner + keep/discard/rollback logic**,
  - the **TSV/JSONL experiment registry**.
- **Borrow:**
  - rerankers from SentenceTransformers/BGE/FlashRank or APIs,
  - metric computation from ranx/ir_measures,
  - datasets from BEIR/ir_datasets,
  - optionally LLM reranking from RankLLM if/when needed.

### The single best 4–6 week MVP

**Exact user:** a retrieval/RAG engineer who can run Python locally and wants to improve reranking quality for a document QA/search system without manually tuning reranking knobs. (Inference; matches the “human writes program.md, agent iterates” workflow.) citeturn1view0turn1view2  

**Exact use case:** **offline reranking improvement for a RAG-style retrieval benchmark** (retrieve evidence passages), using BEIR-like qrels and a frozen evaluator. citeturn3search11turn7search23  

**Exact bounded editable surface:** `rerank_strategy.py` with:
- candidate_k settings,
- fusion settings (RRF constant/weights),
- reranker selection allowlist,
- text truncation rules.

**Exact frozen eval harness:** `prepare.py` + `eval.py`:
- dataset download + cached candidate pool generation,
- deterministic evaluation computing NDCG@10 via `ranx` or `ir_measures`,
- latency/memory measurement.

**Exact primary metric:** `ndcg@10` on dev split. citeturn7search23turn4search26  

**Exact keep/discard rule:** keep iff:
- `ndcg@10` improves by ≥ 0.002 absolute over current best keep **and**
- p95 latency ≤ 1.10× baseline **and**
- peak memory ≤ 1.15× baseline **and**
- recall@100 ≥ baseline − 0.01  
Otherwise discard; on crash/timeout log crash and reset.

**Exact logging format:** untracked `results.tsv` + per-run `runs/<commit>/metrics.json`.

**Exact stack:** Python + `uv` (optional), `ranx` (or `ir_measures`), one reranker backend (FlashRank or a SentenceTransformers cross-encoder). citeturn1view2turn4search2turn5search3turn6search17  

**Weekly milestones (practical):**
- Week 1: frozen dataset loader + candidate cache + baseline run + evaluator prints stable summary lines.
- Week 2: implement the runner + timeout + `results.tsv` logging + git workflow (branch-per-run).
- Week 3: implement allowlisted reranker backends + strategy surface + guardrail metrics.
- Week 4: integrate an agent driver (or manual loop) that proposes changes and executes keep/discard reliably; add run artifacts.
- Week 5: add stability checks (repeat runs / bootstrap), simplify strategy surface, tighten guardrails.
- Week 6: polish demo + documentation; add one additional dataset as a cross-check.

**What NOT to build yet:**
- online A/B testing,
- end-to-end answer generation quality scoring,
- training/fine-tuning new rerankers,
- full retrieval backend integration (OpenSearch/Vespa) unless you already depend on it.

**Concrete demo story:** “Clone repo → run `prepare.py` once → create `autoresearch/<tag>` branch → run the loop overnight → open `results.tsv` in the morning → inspect best kept commit diff in `rerank_strategy.py` → run frozen test eval → merge.”

## Risks, open questions, and final verdict

### Top technical risks

1) **Label leakage / reward hacking**: reranking tasks make it easy to overfit or cheat if qrels are readable; strongest mitigation is sandboxing or hidden test evaluation. (Inference; grounded in general evaluator-hacking risk and the fact you’re optimizing a direct metric.) citeturn1view0turn7search23  
2) **Noisy metrics under small dev sets**: with small query sets, tiny deltas can be noise; you need effect size thresholds and stability checks. citeturn4search2turn7search23  
3) **Latency explosions**: rerank cost scales with candidate count; must enforce time budgets and p95 guardrails. citeturn1view0turn15search1turn3search20  
4) **Nondeterminism in LLM reranking**: API variability and MoE nondeterminism require repeated runs and prompt logging; RankLLM explicitly addresses reliability concerns. citeturn9search1turn9search13  
5) **Scope creep**: once you allow the agent to modify retrieval, chunking, and generation, the loop becomes untrustworthy and debugging becomes research. (Inference consistent with why autoresearch keeps a tiny repo + fixed harness.) citeturn1view2turn1view0  

### Single most important early design decision

**Verdict (inference):** The single most important decision is **what exactly is frozen, and what exactly is editable**—and ensuring that boundary produces experiments that are **comparable across iterations** (same candidate pools, same splits, same metric computation, same budgets). This is explicitly the core idea in Karpathy’s repo structure (frozen `prepare.py`, editable `train.py`) and is what makes the loop meaningful. citeturn1view0turn1view2  

### Final verdict

If you build *exactly* the Karpathy pattern for reranking—**one tiny editable strategy surface, a frozen evaluator that the agent cannot touch, fixed budgets, and auditable git + TSV logging**—this is worth building as a reusable template because it operationalizes “continuous relevance tuning” in a way most stacks do not. If you loosen the boundaries (multiple editable directories, mutable evaluation, unclear keep rules), you’ll end up with a generic “agent tinkers with a search system” project whose gains are hard to trust or reproduce.

## Sources

User-provided project specification (uploaded file). fileciteturn0file0

Key primary references for the autoresearch pattern:
- Karpathy autoresearch README and structure (frozen `prepare.py`, editable `train.py`, `program.md`, fixed 5-minute budget, val_bpb metric). citeturn1view2turn1view0turn1view1  

Reranking / multi-stage ranking fundamentals:
- BERT reranking framing and results. citeturn3search0turn3search20  
- T5-based ranking (MonoT5 / related). citeturn3search1turn3search5  
- ColBERT late interaction (efficiency/effectiveness tradeoff). citeturn3search10turn3search6  

Hybrid retrieval and fusion:
- Reciprocal Rank Fusion original paper + modern implementations/docs. citeturn8search4turn8search7turn8search11turn8search15  

LLM-based reranking:
- RankLLM package and paper; RankVicuna. citeturn9search1turn9search13turn9search2turn9search9  

Evaluation harness/tooling:
- `pytrec_eval`, `ir_measures`, `ranx`, `ir_datasets`. citeturn4search0turn4search1turn4search2turn4search3turn4search28  
- TREC DL metrics guidance emphasizing NDCG@10 and known metric artifacts. citeturn7search23turn7search3turn7search31  

Benchmarks:
- BEIR benchmark and framework. citeturn3search11turn3search3  
- MIRACL multilingual benchmark. citeturn7search0turn7search8  
- MS MARCO and Deep Learning Track framing (full ranking vs reranking). citeturn7search14turn7search11  

Hosted rerank APIs (examples):
- Cohere rerank. citeturn5search0turn5search16  
- Voyage reranker. citeturn5search1turn5search9  
- Jina reranker. citeturn5search2turn5search14  
- Pinecone rerank. citeturn15search2turn15search10turn15search22  

Direct links (most important) — copyable:
```text
https://github.com/karpathy/autoresearch
https://github.com/karpathy/autoresearch/blob/master/program.md

https://arxiv.org/abs/1901.04085
https://arxiv.org/abs/2003.06713
https://arxiv.org/abs/2004.12832

https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking
https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/rrf-retriever

https://github.com/cvangysel/pytrec_eval
https://ir-measur.es/
https://github.com/AmenRa/ranx
https://github.com/allenai/ir_datasets

https://github.com/beir-cellar/beir
https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf

https://arxiv.org/abs/2505.19284
https://github.com/castorini/rank_llm
https://arxiv.org/abs/2309.15088

https://docs.cohere.com/reference/rerank
https://docs.voyageai.com/reference/reranker-api
https://jina.ai/reranker/
https://docs.pinecone.io/reference/api/2025-04/inference/rerank

https://arxiv.org/abs/2310.03714
https://github.com/stanfordnlp/dspy
```