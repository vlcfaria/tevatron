# Dev-set data findings

Last updated: 2026-10-21

This document records the investigation into the checkpoint-validation dev
corpora built from `rlhn/rlhn-680K`, why some groups behave differently from
`Tevatron/beir-corpus`, and why ArguAna currently reports `NDCG@1 = 0`.

## TL;DR

- The dev passages come from the **BGE training collection**, not from
  `Tevatron/beir-corpus`. RLHN is a cleaned subset of the BGE collection.
- `fiqa` matches `Tevatron/beir-corpus` exactly.
- `fever` and `hotpotqa` match after stripping/collapsing whitespace.
- `msmarco_passage` is mostly the same corpus with a title/topic prefix.
- `arguana` is the same underlying arguments with an extra topic prefix and
  occasional first-character loss; because ArguAna is small, the original BEIR
  copies remain in the proxy and outrank the injected RLHN copies.
- `nq` is a granularity mismatch: RLHN passages are longer NQ documents that
  contain/concatenate several BEIR passages.
- `scidocsrr` has no literal overlap with the BEIR SCIDOCS corpus and should be
  dropped from the BEIR-proxy dev set.
- `e5-base-rlhn-680K` was trained on the same RLHN collection used to sample
  the dev queries. Its dev numbers are optimistic/leaky unless training excludes
  the dev rows.

## Build status

The first encoding job was interrupted while encoding the full msmarco corpus.

| group | queries | docs | original | augmented |
|---|---:|---:|---:|---:|
| arguana | 934 | 15,488 | 8,582 | 6,906 |
| fever | 1,000 | 99,668 | 84,801 | 14,867 |
| fiqa | 1,000 | 31,672 | 19,356 | 12,316 |
| hotpotqa | 1,000 | 114,806 | 94,255 | 20,551 |
| msmarco_passage | - | not built | - | - |
| nq | - | not built | - | - |
| scidocsrr | - | not built | - | - |

`encode_interrupt.error` ends during full msmarco corpus encoding at batch
2/8635 with no traceback, consistent with preemption. At that rate it exceeds
the 15h walltime.

`dev-encoding` currently uses about 40 GB, mostly fever (~21 GB) and hotpotqa
(~20 GB). These are full-corpus reference encodings used to build the proxies.

## Where the data comes from

Provenance chain:

```text
cfli/bge-full-data
  -> nthakur/bge-full-data          # Tevatron port
  -> rlhn/default-680K              # 7 retrieval subsets
  -> rlhn/rlhn-680K                 # false negatives cleaned/relabeled
```

Key facts:

- `rlhn/rlhn-680K` is **not a new corpus**. It is a cleaned subset of the BGE
  training collection. The RLHN card says: "RLHN datasets are not new training
  datasets, but rather existing BGE collection training datasets with hard
  negatives cleaned."
- `rlhn/default-1.6M` describes the port from `cfli/bge-full-data`:
  - the whole document is pushed into `text`,
  - `title` is left empty,
  - `docid` is the MD5 hash of the text.
- RLHN relabels false negatives with GPT-4o-mini/GPT-4o. ArguAna is explicitly
  not relabeled: `check_arguana()` in `../rlhn/rlhn/dataset/upload.py` forces
  the `"default"` technique for ArguAna.
- The BGE-M3 paper (`arXiv:2402.03216`) describes BGE data collection:
  - unsupervised data from MTP, S2ORC/Wikipedia, xP3/mC4/CC-News, NLLB/CCMatrix,
    CodeSearchNet;
  - labeled retrieval data from MS MARCO, HotpotQA, NQ, TriviaQA, SQuAD, COLIEE,
    PubMedQA, SimCSE NLI, plus Chinese and multilingual datasets;
  - synthetic MultiLongDoc data.
- The BGE data is public, but the original corpus-building preprocessing is not
  fully public:
  - `cfli/bge-full-data` is public (2.16M rows, 56.7 GB) but its dataset card is
    empty.
  - `nthakur/bge-full-data` documents the Tevatron port, not the original BGE
    filtering/source selection.
  - `FlagEmbedding/dataset` links to the released datasets but does not publish
    the script that assembled `cfli/bge-full-data`.
  - The RLHN curation code is public at `../rlhn` / `github.com/castorini/rlhn`.

The practical consequence: RLHN/BGE passages are only partially the same corpus
as `Tevatron/beir-corpus`. This is a source/preprocessing mismatch, not a
retrieval bug in Tevatron.

## How corpus text is formatted

`EncodeDataset.__getitem__` (`src/tevatron/retriever/dataset.py`) indexes corpus
documents as:

```python
content_text = content.get("text", "")
if "title" in content:
    content_text = content["title"] + " " + content_text
content_text = self.data_args.passage_prefix + content_text.strip()
```

So the indexed text is effectively:

```text
passage_prefix + (title + " " + text).strip()
```

- The title is included whenever the `title` key exists, even if it is empty.
- `passage_field_separator` is currently unused.
- `build_dev_corpus.passage_key()` mirrors this with
  `normalize_passage(title + " " + text)`.

Injected RLHN docs have `title=""` and `text=full passage`, so they are indexed
as `passage_prefix + normalized full passage`. This matches the original BEIR
document only when the BEIR title plus body produces the same text.

## Text-match diagnostics

A hash-based check compared every RLHN positive/negative against the full
`Tevatron/beir-corpus` text.

Positive passages:

| group | raw exact | strip exact | shared first 64 | unmatched |
|---|---:|---:|---:|---:|
| arguana | 0 | 1.1% | 0.8% | 97.9% |
| fever | 0 | 100% | 0 | 0 |
| fiqa | 100% | 0 | 0 | 0 |
| hotpotqa | 0 | 100% | 0 | 0 |
| nq | 0 | 0 | 1.5% | 98.5% |
| scidocsrr | 0 | 0 | 0 | 100% |
| msmarco_passage | 0 | 77.3% | 17.3% | 5.4% |

Negatives follow the same pattern. `docid_match` was 0 everywhere: RLHN uses
MD5 text ids and has no BEIR docid mapping.

Interpretation:

- `fiqa`: exact match; no fix needed.
- `fever`, `hotpotqa`: same text, but raw `passage_key` had an extra leading
  space because RLHN folded the title into `text`. Unconditional strip/NFKC
  fixes the dedup.
- `msmarco_passage`: mostly the same passage with a title/topic prefix; some
  rows are truncated or otherwise different.
- `arguana`: mostly unmatched even after strip; the passages are same-content
  but differently rendered, and the BEIR copies remain in the proxy.
- `nq`: mostly unmatched because RLHN passages are longer documents.
- `scidocsrr`: completely unmatched; different source/preprocessing.

## Similarity examples

The examples below come from
`scripts/check_dev_passage_similarity.py`. The script samples dev passages,
finds candidate corpus docs by literal substring, and reports token Jaccard and
character-level similarity.

### Arguana

RLHN passages are usually the BEIR body with a topic/title prefix prepended.

```text
RLHN: "modern culture traditions house dreaming white christmas junior Snow is actually a lot of fun..."
BEIR: "Snow is actually a lot of fun..."
Jaccard=0.919 char_ratio=0.944 suffix=True
```

```text
RLHN: "human rights international law ethics political philosophy house believes fundamental Fundamental human rights were 'new'..."
BEIR: "Fundamental human rights were 'new'..."
Jaccard=0.920 char_ratio=0.949 suffix=True
```

```text
RLHN: "global house believes sanctions should be used promote democracy Sanctions are ineffective..."
BEIR: "Sanctions are ineffective..."
Jaccard=0.960 char_ratio=0.984 suffix=True
```

So ArguAna RLHN/BEIR pairs are near-duplicates. The BEIR body is often a suffix
of the RLHN text. Some pairs also lose the first character of the title/topic.

### MSMARCO

MSMARCO is even more clearly the same passage with a title/topic prefix.

```text
RLHN: "- Known for its showy 3-5 flowers, the genus Mandevilla..."
BEIR: "Known for its showy 3-5 flowers, the genus Mandevilla..."
Jaccard=1.000 char_ratio=0.995 suffix=True
```

```text
RLHN: "- Common naproxen side effects may include..."
BEIR: "Common naproxen side effects may include..."
Jaccard=1.000 char_ratio=0.995 suffix=True
```

```text
RLHN: "Bruised Ribs Healing Time Bruised ribs indicate an injury..."
BEIR: "Bruised ribs indicate an injury..."
Jaccard=0.983 char_ratio=0.976 suffix=True
```

### NQ

NQ is not a prefix mismatch. RLHN passages are longer NQ documents that contain
one or more BEIR passages.

```text
RLHN: "Canadian Shield bedrock. The North American craton is the bedrock...
       The Canadian Shield is part of an ancient continent called Arctica..."
BEIR doc13869: "The North American craton is the bedrock..."
BEIR doc13870: "The Canadian Shield is part of an ancient continent called Arctica..."
```

`doc13870` scores Jaccard=0.635 and `doc13869` also appears in the RLHN text
(Jaccard=0.270). This is a document-level vs passage-level mismatch, not a
simple formatting difference.

### SciDocsRR

The similarity script found **zero** literal substring candidates across the
sampled passages. The RLHN text also looks noisy (`Availabilay`, `inore`,
`ina`), suggesting a different SciDocs/SciDocsReranking rendering than the BEIR
SCIDOCS corpus. This is not a simple prefix/suffix mismatch. Decision: drop
`scidocsrr` from the BEIR-proxy dev set.

### Caveat about negatives

The similarity script finds the best literal-substring candidate for the passage
text, not the best query-relevant document. Hard negatives that do not exist in
the BEIR corpus can show unrelated "best matches". The positive examples are the
meaningful signal; do not interpret low-similarity negative matches as evidence
about the positive/qrels.

## Why ArguAna has `NDCG@1 = 0`

This is the important failure mode.

`NDCG@1` is effectively `P@1` for binary qrels. Zero means that for every
ArguAna query, the top-ranked document is not in the qrels. At the same time,
`Recall@100` is ~0.97-1.00, so the relevant documents are being retrieved
somewhere in the top 100.

The mechanism:

1. ArguAna has only 8,674 BEIR docs. The reference model's top-100 union covers
   8,582 of them, i.e. almost the whole corpus.
2. `build_dev_corpus` injects the RLHN positives/negatives as `assigned-*`
   documents. Qrels point only to those injected ids.
3. The RLHN texts are not byte-identical to the BEIR texts:
   - they usually prepend the ArguAna topic/title prefix,
   - the BEIR title is sometimes empty,
   - the first character of the title/topic is sometimes lost.
4. Because the texts differ, the exact-text dedup does not remove the original
   BEIR copy. The proxy therefore contains both:
   - the original BEIR doc, which is not in qrels,
   - the injected RLHN copy, which is in qrels.
5. The retriever usually ranks the original BEIR doc first. The original is
   cleaner and semantically identical; the injected copy is not a qrels match.
6. Therefore the top-1 document is a "miss" for essentially every query:
   `NDCG@1 = 0`, `MRR@1 = 0`, while `Recall@100` stays high because the
   injected copy still appears later.

Concrete example from the earlier diagnostic:

```text
qrels positive = assigned-1604:
  "conomic policy tax politics ... It is really not up to the Government..."

original BEIR doc test-economy-eptpghdtre-pro03b:
  title: "economic policy tax politics ..."
  text:  "It is really not up to the Government..."
```

The original BEIR doc was already at rank 3 for that query in the reference
rank list. The model gets no credit for it because qrels only accept the
injected `assigned-*` copy.

This is not an eval-metric bug. It is a qrels/corpus-duplication bug caused by
mixing two different renderings of the same source data.

For compatible groups (`fiqa`, `fever`, `hotpotqa`), the normalized
`passage_key` matches the original BEIR doc after our strip/NFKC change, so the
original is removed and the injected copy is the only copy. That fixes the
dedup and qrels alignment. For ArguAna, normalization alone does not help
because the text differs beyond whitespace/first character.

## Metrics snapshot

These are the metrics present in `output/eval` when the investigation was done.
`e5-base-unsupervised` was still being written, so treat it as partial.

| model | group | NDCG@1 | NDCG@10 | Recall@100 |
|---|---|---:|---:|---:|
| e5-base | arguana | 0.000 | 0.239 | 0.968 |
| e5-base | fever | 0.400 | 0.551 | 0.934 |
| e5-base | fiqa | 0.353 | 0.348 | 0.693 |
| e5-base | hotpotqa | 0.766 | 0.629 | 0.849 |
| e5-base-rlhn-680K | arguana | 0.000 | 0.360 | 0.997 |
| e5-base-rlhn-680K | fever | 0.842 | 0.859 | 0.982 |
| e5-base-rlhn-680K | fiqa | 0.441 | 0.465 | 0.806 |
| e5-base-rlhn-680K | hotpotqa | 0.857 | 0.748 | 0.952 |
| e5-base-unsupervised | arguana | 0.001 | 0.173 | 0.937 |
| e5-base-unsupervised | fever | 0.437 | 0.548 | 0.919 |

Notes:

- `NDCG@1 > NDCG@5` is not a bug by itself. BEIR's NDCG@1 is effectively P@1,
  while NDCG@5 uses an ideal DCG over up to 5 positives.
- `e5-base-rlhn-680K` was trained on `rlhn-680K`, the same collection the dev
  queries are sampled from. Unless training uses `dev-split/train/train` or a
  different held-out source, its dev numbers are not a valid early-stopping
  signal.
- Absolute e5 numbers are lower than public BEIR numbers partly because the
  proxy contains RLHN hard negatives and is not the full BEIR corpus.

## Decisions

- Drop `scidocsrr` from the BEIR-proxy dev set.
- Keep `fiqa`, `fever`, and `hotpotqa`; they are source-compatible.
- Use the new strip/NFKC normalization unconditionally for `passage_key`.
- Use `--positives_only` to skip RLHN hard negatives when building the proxy.
  The sampled BEIR corpus then supplies the negatives.
- `msmarco_passage` is mostly compatible after normalization; it still needs
  prefix/truncation handling for the remaining rows.
- `arguana` and `nq` need either:
  - a source-consistent pooled corpus from the RLHN/BGE passages, or
  - a suffix/containment resolver that maps RLHN passages back to original
    corpus docids, or
  - exclusion from the early-stopping criterion.
- For early stopping, use the compatible groups as the primary signal.

## Implemented changes

- `src/tevatron/utils/eval/build_dev_corpus.py`
  - added `normalize_passage()` (NFKC + whitespace collapse);
  - `passage_key()` now normalizes unconditionally, in both normal and
    positives-only modes;
  - added `--positives_only` to skip dev hard negatives;
  - wired the flag through `main()` and `parse_dev_corpus_args()`.
- `scripts/check_dev_passage_similarity.py`
  - samples positives/negatives from `dev-split/dev/<group>`;
  - finds candidate corpus passages by literal substring;
  - reports best matches with token Jaccard, character ratio, and
    exact/prefix/suffix flags.

## Reproduction commands

Similarity examples:

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python scripts/check_dev_passage_similarity.py --group arguana --n 3

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python scripts/check_dev_passage_similarity.py --group nq --n 3

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python scripts/check_dev_passage_similarity.py --group scidocsrr --n 3

HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python scripts/check_dev_passage_similarity.py --group msmarco_passage --n 3 --max_candidates 2000
```

Rebuild a compatible group with positives only, reusing the existing encoding
cache:

```bash
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
python -m tevatron.utils.eval.build_dev_corpus \
  --model_name_or_path intfloat/e5-base-unsupervised \
  --query_max_len 512 --passage_max_len 512 \
  --pooling mean --normalize true \
  --query_prefix 'query: ' --passage_prefix 'passage: ' \
  --depth 100 --batch_size 64 --shard_search --gpu_float32 \
  --dataset_map ./map.json --encoding_cache_dir ./dev-encoding \
  --output_dir ./dev-corpus-final \
  --positives_only
```

Use a `map.json` containing only the groups you want to rebuild, otherwise the
script will also try to build msmarco/nq/scidocsrr.

## Open limitations

- The similarity script uses literal substring candidate search. A
  `no candidate found` result does not prove the passages are unrelated; it
  means there is no exact 40/80-character overlap. This matters most for NQ and
  SciDocsRR.
- Negative samples that have no BEIR counterpart can show unrelated best
  matches. Do not interpret those as positive/qrels evidence.
- The original BGE corpus-building preprocessing is not fully public, so we
  cannot perfectly reproduce its source selection/truncation.
- ArguAna's exact text relationship is not a single deterministic prefix rule:
  some titles are empty, some prefixes are missing the first character, and
  some passages are truncated differently.
