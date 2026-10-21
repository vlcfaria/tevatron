"""Checkpoint validation on a frozen dev set (DPR/Asyncval-style)

Per group (one dev subset), encode the dev queries and the dev corpus proxy,
flat-IP search, and score against `<eval_corpus_dir>/<group>.qrels.json` with
`tevatron.eval.metrics.score` (full ndcg/map/mrr/recall/precision block).

The expected layout, is, as produced by `tevatron.utils.build_dev_corpus` and `tevatron.utils.sample_dev_queries`:

    <eval_query_dir>/<group>          # save_to_disk tevatron query DatasetDict, 'test' split
    <eval_corpus_dir>/<group>         # save_to_disk tevatron corpus Dataset (docid/title/text)
    <eval_corpus_dir>/<group>.qrels.json

There are two ways to evaluate the models: during training (`sync`) or through a command (`command`), which spawns a command
in every checkpoint save, enabling asynchronous valuation or more complex approaches such as an sbatch job.

  - `sync`    : used from the trainer (--validate_mode sync). Evaluates the
                in-memory weights on every checkpoint save, rank 0, and logs
                the metrics through `trainer.log`. In-memory = the unwrapped
                module; ZeRO-3 params are gathered for the forward, while FSDP
                is not supported

  - `command` : used from the trainer (--validate_mode command). Spawns
                --validate_command on every checkpoint save (e.g. an sbatch job
                on a dedicated GPU).

There is also standalone CLI usage for backfill or pooling approaches, and can be chained with the `command` approach:

    python -m tevatron.retriever.driver.validate \
        --model_name_or_path output/checkpoint-1000 \
        --eval_query_dir dev-split/dev --eval_corpus_dir dev-corpus-final \
        --validate_output_dir output/eval \
        --pooling mean --normalize true \
        '--query_prefix=query: ' '--passage_prefix=passage: ' \
        --query_max_len 64 --passage_max_len 512 \
        --per_device_eval_batch_size 256 --bf16

    # sweep: --validate_checkpoints 'output/checkpoint-*' (then
    # --model_name_or_path is the base model used for the tokenizer)
"""

import glob
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, HfArgumentParser, TrainerCallback

from tevatron.retriever.arguments import (
    DataArguments,
    ModelArguments,
    ValidateArguments,
)
from tevatron.retriever.arguments import (
    TevatronTrainingArguments as TrainingArguments,
)
from tevatron.retriever.driver.encode import encode
from tevatron.retriever.modeling import DenseModel
from tevatron.retriever.searcher import FaissFlatSearcher

logger = logging.getLogger(__name__)

DEFAULT_DEV_K_VALUES = (1, 5, 10, 20, 100, 200)
CHECKPOINT_PREFIX = "checkpoint"


def check_validate_args(validate_args: ValidateArguments):
    """Fail fast on invalid validation args"""
    missing = [
        name
        for name in ("eval_query_dir", "eval_corpus_dir", "validate_output_dir")
        if getattr(validate_args, name) is None
    ]
    if missing:
        raise ValueError(
            f"validation requires {missing} (validate_mode={validate_args.validate_mode})"
        )
    if validate_args.validate_mode == "command" and not validate_args.validate_command:
        raise ValueError("validate_mode=command requires --validate_command")


def discover_groups(eval_query_dir: str, eval_corpus_dir: str) -> list[str]:
    """Groups are query subdirs that also have a corpus dir and qrels file."""
    groups = []

    for group in sorted(os.listdir(eval_query_dir)):
        corpus = os.path.join(eval_corpus_dir, group)
        qrels = os.path.join(eval_corpus_dir, f"{group}.qrels.json")
        if not os.path.isdir(os.path.join(eval_query_dir, group)):
            continue
        if not os.path.isdir(corpus) or not os.path.isfile(qrels):
            logger.warning(
                "[skip] %s — missing corpus dir or %s.qrels.json", group, group
            )
            continue
        groups.append(group)

    if not groups:
        raise FileNotFoundError(
            f"no eval group with corpus+qrels under {eval_query_dir} / {eval_corpus_dir}"
        )

    return groups


def _encode_args(
    data_args: DataArguments,
    validate_args: ValidateArguments,
    name: str,
    group: str,
    is_query: bool,
) -> DataArguments:
    return replace(
        data_args,
        dataset_name=name,
        dataset_config=group,
        dataset_path=None,
        dataset_split="test" if is_query else "train",
        dataset_number_of_shards=1,
        dataset_shard_index=0,
        encode_is_query=is_query,
        encode_output_path=None,
        query_max_len=validate_args.validate_query_max_len or data_args.query_max_len,
        passage_max_len=validate_args.validate_passage_max_len
        or data_args.passage_max_len,
    )


def evaluate_group(
    model, tokenizer, data_args, training_args, validate_args, group
) -> dict:
    from tevatron.eval.metrics import score

    k_values = validate_args.validate_k_values or DEFAULT_DEV_K_VALUES

    q_reps, q_ids = encode(
        model,
        tokenizer,
        _encode_args(
            data_args, validate_args, validate_args.eval_query_dir, group, True
        ),
        training_args,
    )
    p_reps, p_ids = encode(
        model,
        tokenizer,
        _encode_args(
            data_args, validate_args, validate_args.eval_corpus_dir, group, False
        ),
        training_args,
    )

    searcher = FaissFlatSearcher(p_reps)
    searcher.add(p_reps)
    scores, indices = searcher.search(q_reps, min(max(k_values), len(p_ids)))

    # Dev rows can repeat a query_id (same query text); the dict collapses them.
    results = {}
    for qid, row_scores, row_indices in zip(q_ids, scores, indices):
        results[str(qid)] = {
            str(p_ids[i]): float(s) for s, i in zip(row_scores, row_indices)
        }

    with open(os.path.join(validate_args.eval_corpus_dir, f"{group}.qrels.json")) as f:
        qrels = json.load(f)

    missing = [qid for qid in qrels if qid not in results]
    assert not missing, (
        f"{group}: {len(missing)} qrels queries missing from ranklist (e.g. {missing[:5]})"
    )

    return score(qrels, results, k_values=k_values)


def _flatten(metrics: dict, prefix: str) -> dict:
    return {
        f"{prefix}/{family}@{key.split('@')[1]}": v
        for family, values in metrics.items()
        for key, v in values.items()
    }


def _macro_average(blocks: dict) -> dict:
    acc = {}
    for metrics in blocks.values():
        for family, values in metrics.items():
            for key, v in values.items():
                acc.setdefault(family, {}).setdefault(key, []).append(v)
    return {
        family: {key: sum(vs) / len(vs) for key, vs in values.items()}
        for family, values in acc.items()
    }


def evaluate_model(
    model, tokenizer, data_args, training_args, validate_args, name: str
) -> dict:
    """Encode + search + score every subset. Writes JSONs and logs flat metrics for logging."""
    check_validate_args(validate_args)

    logger.info(
        "[validate] %s query_max_len=%d passage_max_len=%d",
        name,
        validate_args.validate_query_max_len or data_args.query_max_len,
        validate_args.validate_passage_max_len or data_args.passage_max_len,
    )

    out_dir = Path(validate_args.validate_output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks, flat = {}, {}
    for group in discover_groups(
        validate_args.eval_query_dir, validate_args.eval_corpus_dir
    ):
        logger.info("[validate] %s / %s", name, group)
        metrics = evaluate_group(
            model, tokenizer, data_args, training_args, validate_args, group
        )
        blocks[group] = metrics
        flat.update(_flatten(metrics, f"eval/{group}"))
        with open(out_dir / f"{group}.metrics.json", "w") as f:
            json.dump({"dataset": group, **metrics}, f, indent=2)

    mean = _macro_average(blocks)
    with open(out_dir / "aggregate.metrics.json", "w") as f:
        json.dump({"name": name, "groups": list(blocks), "mean": mean}, f, indent=2)
    flat.update(_flatten(mean, "eval/mean"))
    return flat


def _step_of(path: str) -> int:
    match = re.search(
        rf"{CHECKPOINT_PREFIX}-(\d+)", os.path.basename(os.path.normpath(path))
    )
    return int(match.group(1)) if match else 0


class _Reporting:
    """Drive HF's reporting integrations (wandb/tensorboard/...) without a Trainer.

    Uses the same `--report_to` as training so the async CLI lands on the same
    dashboards. To attach to the training run, set WANDB_RUN_ID and
    WANDB_RESUME=allow in the eval job's environment.
    """

    def __init__(self, args):
        from transformers.integrations import get_reporting_integration_callbacks
        from transformers.trainer_callback import TrainerState

        self.args = args
        self.state = TrainerState()
        self.callbacks = [
            cls() for cls in get_reporting_integration_callbacks(args.report_to)
        ]

    def log(self, metrics: dict, step: int):
        self.state.global_step = step
        for callback in self.callbacks:
            callback.on_log(self.args, self.state, None, logs=dict(metrics))


class ValidateCallback(TrainerCallback):
    """on_save hook: inline dev eval (`sync`) or spawn `validate_command` (async)."""

    def __init__(
        self, data_args: DataArguments, validate_args: ValidateArguments, tokenizer
    ):
        self.data_args = data_args
        self.validate_args = validate_args
        self.tokenizer = tokenizer
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_save(
        self, args, state, control, model=None, processing_class=None, **kwargs
    ):
        validate_args = self.validate_args
        if validate_args.validate_mode == "off":
            return control

        tokenizer = processing_class or kwargs.get("tokenizer") or self.tokenizer
        name = f"{CHECKPOINT_PREFIX}-{state.global_step}"

        if validate_args.validate_mode == "command":
            if args.should_save:
                cmd = (
                    validate_args.validate_command.replace(
                        "{checkpoint}", os.path.join(args.output_dir, name)
                    )
                    .replace("{step}", str(state.global_step))
                    .replace("{output_dir}", args.output_dir)
                )
                logger.info("[validate] launching async: %s", cmd)
                subprocess.Popen(cmd, shell=True)
            return control

        was_training = model.training
        with self._gather_if_zero3(model):
            if args.should_save:
                try:
                    flat = evaluate_model(
                        model, tokenizer, self.data_args, args, validate_args, name
                    )
                    if self.trainer is not None:
                        self.trainer.log(flat)
                finally:
                    model.train(was_training)

        # Other ranks raced ahead while rank 0 evaluated; sync back up at the
        # next collective anyway, this just makes the pause explicit.
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        return control

    def _gather_if_zero3(self, model):
        """Gather ZeRO-3 partitions for the eval forward

        All ranks must enter (it is a collective); modifier_rank=0 keeps the
        gathered copy on rank 0 only. No-op for non-DeepSpeed and ZeRO 0/1/2
        (those params have no ds_id, so the context filters them out).
        """
        from contextlib import nullcontext

        if self.trainer is not None and getattr(
            self.trainer, "is_deepspeed_enabled", False
        ):
            import deepspeed

            return deepspeed.zero.GatheredParameters(
                list(model.parameters()), modifier_rank=0
            )
        return nullcontext()


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ValidateArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, validate_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, validate_args = (
            parser.parse_args_into_dataclasses()
        )

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
    )
    check_validate_args(validate_args)

    if validate_args.validate_checkpoints:
        checkpoints = sorted(
            glob.glob(validate_args.validate_checkpoints), key=_step_of
        )
    else:
        checkpoints = [model_args.model_name_or_path]

    if not checkpoints:
        raise FileNotFoundError(
            f"no checkpoints matched {validate_args.validate_checkpoints}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = (
        torch.bfloat16
        if training_args.bf16
        else torch.float16
        if training_args.fp16
        else torch.float32
    )
    reporter = _Reporting(training_args)

    for checkpoint in checkpoints:
        name = os.path.basename(os.path.normpath(checkpoint))
        marker = (
            Path(validate_args.validate_output_dir) / name / "aggregate.metrics.json"
        )
        if marker.exists() and not validate_args.validate_overwrite:
            logger.info("[skip] %s — already evaluated", name)
            continue

        logger.info("[validate] loading %s", checkpoint)
        model = DenseModel.load(
            checkpoint,
            pooling=model_args.pooling,
            normalize=model_args.normalize,
            lora_name_or_path=model_args.lora_name_or_path,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch_dtype,
            attn_implementation=model_args.attn_implementation,
        )
        flat = evaluate_model(
            model, tokenizer, data_args, training_args, validate_args, name
        )
        reporter.log(flat, _step_of(checkpoint))
        logger.info(
            "[validate] %s mean NDCG@10=%.4f",
            name,
            flat.get("eval/mean/ndcg@10", float("nan")),
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
