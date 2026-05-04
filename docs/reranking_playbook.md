# Reranking Playbook for the Autoresearch Controller

Purpose: give a small local controller enough structured knowledge to propose benchmark-aware reranking experiments without live web access.

Scope: this playbook is for the inner autoresearch loop. It is not a general IR textbook. It is meant to help the controller choose one bounded change to `rerank_strategy.py` and avoid repeating weak ideas.

---

## 1. Operating rules

1. Optimize `ndcg@10` on the frozen dev split.
2. Respect guardrails on `recall@100`, `latency_p95_ms`, run validity, and any local cost or memory limits.
3. Prefer mechanism changes over tiny parameter nudges.
4. Do not repeat a recent failed family unless the mechanism is materially different.
5. Never propose changes to the eval harness, dataset, metrics, logs, or driver.
6. Use this playbook plus recent run history. Do not rely on live web search inside the loop.

---

## 2. Fast mental model

- First-stage retrieval finds a candidate set.
- Reranking improves the order of that candidate set.
- If recall in the candidate set is weak, reranking cannot recover missing relevant items.
- Cross-encoders are usually stronger but slower, so they are best used on a bounded top-k candidate pool.
- Dev-only wins can fail promotion when a change overfits one dataset mix or one query style.

---

## 3. Proposal families

Each proposal must belong to exactly one primary family.

### Family A: `candidate_k`

**What it changes**
- Number of retrieved candidates sent into the reranker.

**Why it can help**
- Too small: relevant docs never reach the reranker.
- Too large: more compute, more noise, worse latency.

**When to try**
- Recall guardrail is weak.
- Promotion failures suggest the dev loop is overfitting a too-small candidate pool.
- Latency budget has spare room.

**Likely effect**
- `ndcg@10`: up or flat
- `recall@100`: up or flat
- latency: up

**Promotion risk**
- Medium. Small dev gains may disappear on broader benchmarks.

**Good experiments**
- Fixed increase or decrease with a clear justification.
- Query-conditional `candidate_k` based on cheap heuristics like query length.

**Bad experiments**
- Repeating 10 -> 12 -> 14 -> 15 without a new hypothesis.

---

### Family B: `score_normalization`

**What it changes**
- Scale handling before combining retrieval and reranker scores.

**Why it can help**
- Raw scores from different components may not be directly comparable.

**When to try**
- You are already using fusion.
- Different models or retrieval channels produce scores on different scales.

**Likely effect**
- `ndcg@10`: up or flat
- `recall@100`: flat
- latency: flat or slightly up

**Promotion risk**
- Medium to high. Easy to get shallow dev wins.

**Good experiments**
- Switch among a small approved set such as `none`, `zscore`, `minmax`, `rank_only`.

**Bad experiments**
- Endless tiny weighting changes with no mechanism change.

---

### Family C: `fusion_weight`

**What it changes**
- Relative weight between first-stage retrieval score and reranker score.

**Why it can help**
- A pure reranker score may wash out useful retrieval priors.
- Too much retrieval weight can blunt the reranker.

**When to try**
- First-stage retrieval is reasonably good.
- Recent failures were not already repeated fusion tweaks.

**Likely effect**
- `ndcg@10`: up or flat
- `recall@100`: flat
- latency: flat

**Promotion risk**
- High if overused. This is the easiest family for the controller to over-repeat.

**Good experiments**
- Change the fusion rule, not just the number.
- Example: `rank_only` fusion instead of raw score sum.

**Bad experiments**
- Alpha 0.30 -> 0.35 -> 0.40 with identical reasoning.

---

### Family D: `metadata_boost`

**What it changes**
- Lightweight boosts using structured fields such as title, source, doc type, recency, or entity matches.

**Why it can help**
- Some queries are strongly title- or entity-oriented.

**When to try**
- Corpus has reliable metadata.
- Queries often mention entities, titles, or specific phrases.

**Likely effect**
- `ndcg@10`: up or flat
- `recall@100`: flat
- latency: slightly up or flat

**Promotion risk**
- Medium. Can help some datasets and hurt others.

**Good experiments**
- Only apply boosts under cheap query heuristics.
- Keep boosts small and transparent.

**Bad experiments**
- Huge unconditional metadata priors.

---

### Family E: `dedup_filter`

**What it changes**
- Removes near-duplicate candidates or passage clones before or after reranking.

**Why it can help**
- Duplicate passages can crowd out diverse relevant items in the top ranks.

**When to try**
- Candidate pool often contains near-identical text.
- Top ranks contain obvious duplicates.

**Likely effect**
- `ndcg@10`: up or flat
- `recall@100`: flat or slightly down
- latency: slightly up or flat

**Promotion risk**
- Low to medium if dedup is conservative.

**Good experiments**
- Conservative dedup based on normalized title or simple text fingerprint.

**Bad experiments**
- Aggressive dedup that removes semantically distinct but similar-looking passages.

---

### Family F: `truncation_policy`

**What it changes**
- How much text from each document is fed to the reranker.

**Why it can help**
- Long noisy documents may drown useful signal.
- Shorter truncation can reduce cost and latency.

**When to try**
- Corpus documents vary a lot in length.
- Latency pressure is high.

**Likely effect**
- `ndcg@10`: up or down
- `recall@100`: flat
- latency: often down

**Promotion risk**
- Medium. Bad truncation can silently remove critical evidence.

**Good experiments**
- Switch among a small approved set of window sizes.
- Use title + leading chunk patterns.

**Bad experiments**
- Arbitrary truncation changes with no rationale tied to corpus structure.

---

### Family G: `pre_rerank_filter`

**What it changes**
- Cheap filtering before the expensive reranker, such as dropping very low lexical overlap items or unsupported doc types.

**Why it can help**
- Reduces reranker work and noise.

**When to try**
- Candidate pool contains many obviously weak items.
- Latency is under pressure.

**Likely effect**
- `ndcg@10`: up, flat, or down
- `recall@100`: down risk
- latency: down

**Promotion risk**
- Medium to high. Easy to harm recall.

**Good experiments**
- Conservative filters with explicit recall risk assessment.

**Bad experiments**
- Hard pruning with no evidence.

---

### Family H: `query_type_heuristic`

**What it changes**
- Simple query routing inside the strategy, such as treating very short entity queries differently from descriptive natural-language queries.

**Why it can help**
- Different query types may benefit from different `candidate_k`, metadata boosts, or fusion rules.

**When to try**
- Benchmark mix includes clearly different query styles.
- Recent global changes help some queries and hurt others.

**Likely effect**
- `ndcg@10`: up or flat
- `recall@100`: flat or slightly up
- latency: slightly up

**Promotion risk**
- Medium. Bad heuristics can overfit one dev mix.

**Good experiments**
- Use cheap, interpretable rules like query length, punctuation, or named-entity shape.

**Bad experiments**
- Complex classifier logic in v1.

---

## 4. Family selection priorities

Use these priorities when choosing the next experiment.

### Search curriculum

Run the loop in two phases:

1. **Explore**
   - Give each active family one clean attempt before repeatedly tuning one family.
   - Prefer interpretable families first so the loop learns useful failure modes quickly.
2. **Exploit**
   - After first-pass coverage, rank families by measured signal on fast and promotion benchmarks.
   - Downselect to the top 1 to 2 families and tune only those until the signal collapses or cooldown triggers.

### Prefer first
- `candidate_k`
- `truncation_policy`
- `query_type_heuristic`
- `metadata_boost`

### Use carefully
- `score_normalization`
- `fusion_weight`
- `dedup_filter`
- `pre_rerank_filter`

### Delay until later or milestone-only
- none in this phase

Reason: the first group is more interpretable and more likely to produce mechanism-level changes instead of endless score-massage.

---

## 5. Benchmark awareness cards

These cards are intentionally short. The controller should use the repo's actual manifests and past results for specifics.

### Fast loop benchmark card

**Role**
- Cheap local iteration.

**Risk**
- Small wins may be noise or may overfit the dataset mix.

**What to do**
- Accept only clear wins.
- Be suspicious of tiny gains from score-massage families.

### Promotion benchmark card

**Role**
- Check whether a dev win survives on a larger or different benchmark.

**Risk**
- More latency and compute.

**What to do**
- Run only for kept candidates.
- If dev wins fail promotion, reduce trust in that family.

---

## 6. Anti-patterns

Do not propose these without a very strong benchmark-specific reason.

1. Tiny alpha changes after recent fusion failures.
2. Another score-normalization tweak with no new mechanism.
3. Bigger `candidate_k` when latency is already near the ceiling.
4. Hard filters when recall is already fragile.
5. Metadata priors with no reliable metadata.
6. Any hidden checkpoint swap in the daily inner loop.
7. Any change that quietly expands the effective search space beyond the trust boundary.

---

## 7. Novelty policy

Before execution, a proposal should be rejected if all of the following are true relative to a recent failed run:

- same family
- same changed keys or same mechanism slot
- same expected tradeoff profile
- only tiny numeric differences

A proposal should be favored if it is:

- a new family not tried recently
- a materially different mechanism
- motivated by a known failure pattern
- interpretable enough to explain a promotion failure if it happens

Cooldown rule suggestion:
- no same-family proposals back to back
- if one family has 3 discards in the last 5 attempts, cool it down for the next 5 attempts

---

## 8. Proposal schema

Every proposal should fill this structure before it can run.

```json
{
  "family": "candidate_k",
  "hypothesis": "Increasing candidate_k from 10 to 25 should improve top-rank quality on descriptive queries by giving the cross-encoder more relevant options.",
  "changed_keys": ["candidate_k"],
  "why_not_duplicate": "Recent failures were fusion-weight changes. This change expands the reranker search window instead of reweighting scores.",
  "expected_ndcg_direction": "up",
  "expected_recall_direction": "flat",
  "expected_latency_direction": "up",
  "promotion_risk": "medium"
}
```

If the controller cannot fill this cleanly, reject the proposal.

---

## 9. Controller prompt skeleton

Use this as a local prompt basis.

```text
You are the proposal controller for a bounded autoresearch loop for reranking.

Your job is to propose exactly one change to rerank_strategy.py.

Constraints:
- You may only change the bounded strategy surface.
- You may not change eval code, datasets, metrics, logs, or driver logic.
- Optimize ndcg@10 on frozen dev while respecting recall@100 and latency_p95_ms guardrails.
- Do not repeat a recent failed family unless your mechanism is materially different.
- Prefer interpretable mechanism changes over tiny parameter nudges.

Available families:
- candidate_k
- score_normalization
- fusion_weight
- metadata_boost
- dedup_filter
- truncation_policy
- pre_rerank_filter
- query_type_heuristic

Return a structured proposal using the required schema.
```

---

## 10. Reviewer prompt skeleton

```text
You are the reviewer for the autoresearch proposal queue.

Given 3 to 5 candidate proposals, select one.

Reject proposals that:
- duplicate a recent failed family or mechanism
- are generic score-massage ideas with no new mechanism
- are likely to violate latency or recall guardrails
- are hard to interpret if promotion fails

Prefer proposals that:
- explore a new family during the explore phase
- stay within the currently downselected families during the exploit phase
- are benchmark-aware
- are small enough to trust
- teach the system something even if they fail

Return:
- selected proposal id
- reason for selection
- one-sentence risk statement
```

---

## 11. Minimal outer-loop research policy

Do not give the inner loop live web access.

Instead:
- periodically refresh this playbook outside the loop
- add new strategy cards only after human review
- record what changed in the playbook so experiment history remains interpretable

---

## 12. Good first experiment menu

If the loop is currently collapsing into fusion-weight ideas, try these next in roughly this order:

1. `candidate_k` with a real step change
2. `truncation_policy` tied to document length
3. small `metadata_boost` under a cheap query heuristic
4. `query_type_heuristic` that switches between two already-approved settings
5. conservative `dedup_filter` only after the loop has baseline coverage on the first four families

Avoid another fusion-weight run unless recent evidence strongly points there.

---

## 13. Short glossary

- **candidate pool**: items retrieved before reranking
- **reranker**: a stronger but slower model that reorders the candidate pool
- **promotion**: validating a kept dev win on a larger or different benchmark
- **family**: the primary mechanism class of a proposal
- **duplicate proposal**: same family plus same mechanism plus tiny numeric changes
- **false positive**: dev improvement that fails promotion or repeated evaluation
