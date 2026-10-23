# Checkpoint validation (dev-set evaluation during training)

Tevatron can evaluate dense-retriever **checkpoints on a frozen held-out dev
set** while training, DPR/Asyncval-style. Each dev group is scored on a small
corpus proxy: encode dev queries + corpus with the current weights and score against 
dev qrels with the same metric core as the offlineevaluator ([`tevatron.eval.metrics`](../src/tevatron/eval/metrics.py)).

Everything is implemented on Tevatron's own encode/search/score primitives:

| step | reused symbol |
|---|---|
| encode | [`tevatron.retriever.driver.encode.encode`](../src/tevatron/retriever/driver/encode.py) |
| index + search | [`tevatron.retriever.searcher.FaissFlatSearcher`](../src/tevatron/retriever/searcher.py) |
| metrics | [`tevatron.eval.metrics.score`](../src/tevatron/eval/metrics.py) |
| evaluator | [`tevatron.retriever.checkpoint_validation`](../src/tevatron/retriever/checkpoint_validation.py) |
| CLI | [`tevatron.retriever.driver.validate`](../src/tevatron/retriever/driver/validate.py) |
| trainer hook | [`tevatron.retriever.validate_callback`](../src/tevatron/retriever/validate_callback.py) |

There are two delivery modes, backed by the same evaluator:

- **`sync`** — evaluates the in-memory weights on every checkpoint save. Training
  pauses for the duration of the eval (~seconds at 1k queries × ~15k docs).
- **`command`** — on every checkpoint save, spawns a configurable command (e.g.
  an `sbatch` job on a dedicated GPU). Training never stalls; Tevatron stays
  scheduler-agnostic.

## Prerequisites: build the dev set

Validation consumes the artifacts built by two utilities. First sample held-out
queries from a Tevatron-format training dataset:

```bash
HF_DATASETS_OFFLINE=1 python -m tevatron.utils.sample_dev_queries \
  --dataset_name rlhn/rlhn-100k --dataset_split train \
  --group_by column --group_column subset \
  --num_queries 1000 --seed 1234 --output_dir ./dev-split
```

Then build a per-group corpus proxy (top-k corpus passages for the dev queries
plus the dev positives/negatives), which also writes the qrels:

```bash
python -m tevatron.utils.build_dev_corpus \
  --model_name_or_path intfloat/e5-base-unsupervised \
  --query_max_len 512 --passage_max_len 512 --pooling mean --normalize true \
  '--query_prefix=query: ' '--passage_prefix=passage: ' \
  --depth 1000 --batch_size 64 --shard_search --gpu_float32 \
  --dataset_map ./map.json --encoding_cache_dir ./dev-encoding \
  --output_dir ./dev-corpus-final
```

Resulting layout (one directory per group):

```
dev-split/dev/<group>                   # save_to_disk DatasetDict, 'test' split
dev-corpus-final/<group>                # save_to_disk Dataset (docid/title/text)
dev-corpus-final/<group>.qrels.json     # {query_id: {docid: relevance}}
```

Groups are auto-discovered as entries present in both directories. Groups with a
missing corpus or qrels are skipped with a warning, so subsets can be built
incrementally. Train queries are sampled *out* of the training set, so do not
train on the same rows.

## Configuration

Validation is configured by `ValidateArguments`
([`arguments.py`](../src/tevatron/retriever/arguments.py)):

| flag | default | meaning |
|---|---|---|
| `--validate_mode` | `off` | `off`, `sync`, or `command` |
| `--eval_query_dir` | `None` (required) | directory of dev query groups |
| `--eval_corpus_dir` | `None` (required) | directory of dev corpus proxies + `*.qrels.json` |
| `--validate_output_dir` | `None` (required) | metrics output root |
| `--validate_k_values` | `1 5 10 20 100` | cutoffs for all metric families |
| `--validate_query_max_len` | training `--query_max_len` | inference query length |
| `--validate_passage_max_len` | training `--passage_max_len` | inference passage length |
| `--validate_command` | `None` | command template for `command` mode |
| `--validate_checkpoints` | `None` | CLI only: glob of checkpoints to sweep |
| `--validate_overwrite` | `False` | re-score checkpoints that already have metrics |

The model configuration (`--pooling`, `--normalize`, `--query_prefix`,
`--passage_prefix`, max lengths) **must match training** — silently evaluating
with the wrong pooling/normalize is the classic Asyncval footgun.

It is common to train on tight sequences and evaluate on longer ones; that's what
`--validate_query_max_len` / `--validate_passage_max_len` are for.

## Synchronous mode

Add the flags to `tevatron.retriever.driver.train`:

```bash
torchrun --nproc_per_node=8 -m tevatron.retriever.driver.train \
  --output_dir runs/qwen-rlhn --dataset_name json \
  --dataset_path data/rlhn/train.jsonl \
  --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
  --pooling last --normalize true \
  --query_prefix 'query: ' --passage_prefix 'passage: ' \
  --query_max_len 32 --passage_max_len 128 \
  --save_steps 1000 --report_to wandb \
  --validate_mode sync \
  --eval_query_dir dev-split/dev --eval_corpus_dir dev-corpus-final \
  --validate_output_dir runs/qwen-rlhn/eval \
  --validate_query_max_len 64 --validate_passage_max_len 512
```

On `on_save`, rank 0 evaluates the **in-memory** weights (no checkpoint reload),
logs the full metric block through `trainer.log` (so it lands in wandb /
TensorBoard / etc. at the saved step), writes the JSONs, and restores train mode.
Other ranks wait at a barrier while rank 0 evaluates.

Requirements and limits:

- `beir` and `pytrec_eval` must be importable in the *training* environment
  (scoring is imported lazily, so training without validation never needs them).
- DDP and DeepSpeed work. Under ZeRO-3 the parameters are gathered for the eval
  forward (`deepspeed.zero.GatheredParameters`, rank 0 only) and released after.
- FSDP is **not** supported in `sync` mode (params are gathered per-layer via
  collectives, so a rank-0-only forward is invalid). Use `command` mode instead.

## Asynchronous mode (Asyncval-style)

Training only spawns a command per save; the command runs the standalone
evaluator however the cluster requires. Placeholders: `{checkpoint}`, `{step}`,
`{output_dir}`.

```bash
torchrun --nproc_per_node=8 -m tevatron.retriever.driver.train \
  ... \
  --save_steps 1000 \
  --validate_mode command \
  --eval_query_dir dev-split/dev --eval_corpus_dir dev-corpus-final \
  --validate_output_dir runs/qwen-rlhn/eval \
  --validate_command 'sbatch eval-ckpt.sbatch {checkpoint}'
```

`eval-ckpt.sbatch` (consumer-side, not in Tevatron) evaluates one checkpoint on a
dedicated GPU; it is just the standalone CLI:

```bash
python -m tevatron.retriever.driver.validate \
  --model_name_or_path "$1" \
  --eval_query_dir dev-split/dev --eval_corpus_dir dev-corpus-final \
  --validate_output_dir runs/qwen-rlhn/eval \
  --pooling last --normalize true \
  --query_prefix 'query: ' --passage_prefix 'passage: ' \
  --validate_query_max_len 64 --validate_passage_max_len 512 \
  --per_device_eval_batch_size 256 --bf16 --report_to wandb
```

The command is fire-and-forget (`subprocess.Popen`), so `sbatch` returns
immediately. Evaluation is **idempotent**: checkpoints whose
`aggregate.metrics.json` already exists are skipped (`--validate_overwrite` to
force). That makes a polling loop safe without any watcher code:

```bash
for _ in $(seq 1 240); do
  python -m tevatron.retriever.driver.validate \
    --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
    --validate_checkpoints 'runs/qwen-rlhn/checkpoint-*' \
    ... ; sleep 300
done
```

When sweeping, `--model_name_or_path` is only the tokenizer source; weights come
from each checkpoint directory.

### Metrics and wandb

`evaluate_model` returns flat keys, all logged at the checkpoint step:

- `eval/<group>/<metric>@<k>` per group
- `eval/mean/<metric>@<k>` macro-averaged over groups

`<metric>` is one of `ndcg`, `map`, `mrr`, `recall`, `precision`. In `sync` mode
this goes through the trainer's integrations (pass `--report_to wandb`). The
standalone CLI logs through the same Hugging Face integration callbacks using
its own `--report_to`; to attach the async job to the training run, export the
run identity before launching it:

```bash
export WANDB_PROJECT=<project>
export WANDB_RUN_ID=<training-run-id>
export WANDB_RESUME=allow
```

## Output layout

```
<validate_output_dir>/<checkpoint-N>/
  <group>.metrics.json      # {"dataset": group, <full metrics block>}
  aggregate.metrics.json    # {"name": ..., "groups": [...], "mean": <full block>}
```

`sync` mode names the run `checkpoint-<global_step>` to match the async layout,
so metrics from both modes are directly comparable. As with the rest of
Tevatron's evaluation, per-query scoring uses `pytrec_eval`; relevance grades in
the qrels (e.g. distillation `score` fields) are preserved as graded judgments.

## Notes

- Only dense retrievers are covered (SPLADE / multi-vector / rerankers are out of
  scope for now).
