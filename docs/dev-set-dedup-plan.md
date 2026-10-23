# Dev-set cross-source dedup plan

Last updated: 2026-10-21

## Goal

Add a **general, per-dataset dedup mechanism** to `build_dev_corpus` so dev
passages from a source training dataset (RLHN, BGE, or anything else) can be
reconciled with the target retrieval corpus.

The dedup logic must be RLHN-free. Do not hard-code RLHN-specific assumptions
such as:

- RLHN prepends a BGE topic/title prefix;
- RLHN sometimes loses the first character;
- RLHN docids are MD5 hashes.

Those are explanations for the ArguAna failure, not general rules. The general
rules are normalized text equality, prefix/suffix relations, containment, token
similarity, and optionally semantic similarity.

## TLDR: generic methods

| method | generic? | use when | notes |
|---|---|---|---|
| Exact normalized | yes | texts should be identical | current default; NFKC + strip |
| Prefix / suffix | yes | one text is a truncated or prefixed version of the other | cheap, deterministic; add min length/ratio guards |
| Containment | yes | one passage is embedded inside a longer document | needed for NQ-style document/passage mismatch |
| Token Jaccard / MinHash | yes | near-duplicates with formatting/punctuation differences | generic, scalable with candidate blocking/LSH |
| Semantic embedding NN | yes | paraphrases or different wording | expensive; high false-positive risk; opt-in fallback only |
| Cross-encoder/LLM verifier | yes | ambiguous lexical candidates | high precision, expensive; verify only top candidates |
| Source-consistent corpus | yes | source texts are fundamentally different | avoids cross-source dedup entirely |

For the first implementation, support `exact`, `prefix`, `suffix`,
`containment`, and `jaccard`. Defer semantic matching and the LLM verifier until
lexical methods are insufficient.

## Per-dataset configuration

Extend each entry in `map.json` with an optional `dedup` block.

### Suffix example (ArguAna)

```json
{
  "name": "arguana",
  "encode_corpus_config": "arguana",
  "encode_corpus_name": "Tevatron/beir-corpus",
  "query_corpus_config": "arguana",
  "query_corpus_name": "dev-split/dev",
  "dedup": {
    "method": "suffix",
    "action": "drop_original",
    "min_chars": 100,
    "min_ratio": 0.5
  }
}
```

### Other methods

```json
"dedup": {"method": "exact"}
"dedup": {"method": "prefix", "min_chars": 100, "min_ratio": 0.5}
"dedup": {"method": "containment", "min_chars": 100}
"dedup": {"method": "jaccard", "threshold": 0.9, "min_tokens": 20}
```

Absent `dedup` means `exact`, preserving current behavior.

### Proposed schema

```text
dedup.method:      exact | prefix | suffix | containment | jaccard
dedup.action:      drop_original | map_to_original   (default: drop_original)
dedup.min_chars:   minimum length of the shorter text for prefix/suffix/containment
dedup.min_ratio:   minimum len(shorter) / len(longer)
dedup.threshold:   Jaccard threshold for token matching
dedup.min_tokens:  minimum token count for Jaccard
```

Future opt-in additions:

```text
dedup.method: semantic
dedup.model_name_or_path: ...
dedup.threshold: ...
```

Semantic matching must use an independent fixed model, not one of the models
being evaluated.

## Action choice

### `drop_original` (recommended first)

When a dev passage matches a corpus doc, remove the corpus doc and keep the
injected dev copy. Qrels already point to the injected copy.

Pros:

- smaller code change;
- qrels do not need to be rewritten;
- works with one-to-many containment (remove all matched originals).

Cons:

- keeps the dev-source text, which may be longer or differently formatted;
- removes original corpus docs that could be candidates for other queries.

### `map_to_original` (later)

When a dev passage matches a corpus doc, point qrels at the original corpus
docid and skip injecting the dev copy.

Pros:

- cleaner corpus;
- qrels use the original, curated corpus text.

Cons:

- requires rewriting qrels;
- one-to-many matches need an explicit policy;
- a wrong match silently corrupts relevance labels.

Recommendation: implement `drop_original` first, then add `map_to_original` only
if the corpus quality matters enough to justify the extra risk.

## Implementation plan

### 1. Extend dataset configuration

- Add an optional `dedup` field to `DatasetInfo`.
- Parse the nested config in `parse_dev_corpus_args`.
- Validate method names and numeric fields early.
- Keep absent `dedup` equivalent to `{"method": "exact"}`.

### 2. Build a resolver

Create a small resolver abstraction in `build_dev_corpus`:

```text
build_resolver(dev_texts, config) -> resolver
resolver.matches(corpus_key) -> bool
```

For `drop_original`, only a boolean match is needed.

Method implementations:

- **exact**
  - normalize with `normalize_passage`;
  - set membership.
- **prefix / suffix**
  - build a fingerprint index from dev passage keys using the first/last N
    characters;
  - for each corpus key, look up candidates by the same fingerprint;
  - verify full prefix/suffix with `min_chars` and `min_ratio` guards.
- **containment**
  - build a candidate index from distinctive dev substrings or token shingles;
  - verify one string is contained in the other;
  - allow one-to-many matches.
- **jaccard**
  - tokenize both sides;
  - use a rare-token or shingle candidate prefilter;
  - require `len(tokens) >= min_tokens` and Jaccard >= threshold.

Do not compare every corpus doc against every dev passage. Use fingerprints or
an inverted index for candidate generation.

### 3. Apply in `build_dev_corpus`

Replace the exact-only filter:

```python
filtered_corpus = corpus.encode_data.filter(
    lambda x: (
        str(x["docid"]) in docs_set
        and x["text"] not in assigned_map
        and passage_key(x) not in assigned_map
    )
)
```

with the resolver:

```python
filtered_corpus = corpus.encode_data.filter(
    lambda x: (
        str(x["docid"]) in docs_set
        and not resolver.matches(passage_key(x))
    )
)
```

Keep the existing exact key checks inside the `exact` resolver so behavior is
unchanged when `dedup` is absent.

### 4. Log dedup statistics

For each group, print:

- method/action;
- number of original corpus docs before and after dedup;
- number of matched corpus docs;
- number of unmatched dev passages;
- a few matched examples with docid, match type, and score.

This is essential for trusting a threshold-based method.

### 5. Add a dry-run mode

Add a CLI flag such as `--dedup_dry_run` that:

- computes matches;
- prints statistics and examples;
- does not write a new corpus or qrels.

Use it before rebuilding a group.

### 6. Tests

Add small synthetic tests for the resolver:

- exact match;
- normalized exact match;
- suffix match with prefix;
- prefix match with truncation;
- containment;
- Jaccard above/below threshold;
- `min_chars`/`min_ratio` rejection of short/generic matches.

No full-corpus test is needed for the unit-level behavior.

## Rollout one dataset at a time

Use a temporary `map.json` containing only the group being rebuilt. The existing
encoding/search caches in `dev-encoding` are reused when present.

### ArguAna

```json
"dedup": {
  "method": "suffix",
  "action": "drop_original",
  "min_chars": 100,
  "min_ratio": 0.5
}
```

Verify after rebuild:

- original BEIR copies are removed when a suffix match exists;
- qrels still reference injected `assigned-*` docs;
- `NDCG@1` becomes non-zero;
- `NDCG@10`/`Recall@100` move toward plausible ArguAna values.

If suffix coverage is too low, add `jaccard` as a fallback in a later step.

### NQ

Start with `containment`, action `drop_original`:

```json
"dedup": {"method": "containment", "action": "drop_original", "min_chars": 100}
```

NQ needs a one-to-many policy if we ever move to `map_to_original`: one RLHN
document can contain several BEIR passages. For now, `drop_original` avoids
rewriting qrels.

### MS MARCO

Start with exact/strip. If duplicates remain, add:

```json
"dedup": {"method": "prefix", "action": "drop_original", "min_chars": 40, "min_ratio": 0.5}
```

Use Jaccard only if prefix/suffix coverage is insufficient.

### FiQA / FEVER / HotpotQA

Keep the default exact/strip path. They already match after normalization.

### SciDocsRR

Drop from the BEIR-proxy dev set. No dedup method is planned for it.

## Non-goals

- No RLHN-specific title-prefix or first-character heuristics.
- No semantic dedup in the first pass.
- No silent qrels rewriting without match logging.
- No change to the source corpus itself.

## Open decisions

1. Do we start with `drop_original` only, or implement `map_to_original` in the
   same change?
2. For NQ containment, should a merged dev passage map to one original doc or
   several if `map_to_original` is added later?
3. Do we want `jaccard` in the first implementation, or only prefix/suffix/
   containment?
4. Should a failed match on a **positive** passage fail the build, warn, or be
   counted in the stats only?
