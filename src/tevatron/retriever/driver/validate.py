"""Checkpoint validation on a frozen dev set (DPR/Asyncval-style)

Per group (one dev subset), encode the dev queries and the dev corpus proxy,
flat-IP search, and score against `<eval_corpus_dir>/<group>.qrels.json` with
`tevatron.eval.metrics.score` (full ndcg/map/mrr/recall/precision block).

The expected layout, is, as produced by `tevatron.utils.eval.build_dev_corpus` and `tevatron.utils.eval.sample_dev_queries`:

    <eval_query_dir>/<group>          # save_to_disk tevatron query DatasetDict, 'test' split
    <eval_corpus_dir>/<group>         # save_to_disk tevatron corpus Dataset (docid/title/text)
    <eval_corpus_dir>/<group>.qrels.json

There are two ways to evaluate the models: during training (`sync`) or through a command (`command`), which spawns a command
in every checkpoint save, enabling asynchronous valuation or more complex approaches such as an sbatch job.

  - `sync`    : used from the trainer (--validate_mode sync). Evaluates the
                in-memory weights on every checkpoint save, rank 0, and logs
                the metrics through `trainer.log`. ZeRO-3 params are gathered
                for the forward, while FSDP is not supported

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

    # for sweeping multiple checkpoints: --validate_checkpoints 'output/checkpoint-*' (then
    # --model_name_or_path is the base model used for the tokenizer)
"""

import glob
import logging
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, HfArgumentParser

from tevatron.retriever.arguments import (
    DataArguments,
    ModelArguments,
    ValidateArguments,
)
from tevatron.retriever.arguments import (
    TevatronTrainingArguments as TrainingArguments,
)
from tevatron.retriever.checkpoint_validation import (
    CHECKPOINT_PREFIX,
    check_validate_args,
    evaluate_model,
)
from tevatron.retriever.modeling import DenseModel
from tevatron.retriever.validate_callback import ValidateCallback

__all__ = ["ValidateCallback", "check_validate_args", "main"]

logger = logging.getLogger(__name__)


def _step_of(path: str) -> int:
    match = re.search(
        rf"{CHECKPOINT_PREFIX}-(\d+)", os.path.basename(os.path.normpath(path))
    )
    return int(match.group(1)) if match else 0


class _Reporter:
    """Forward metrics to the configured HF reporting integrations without a Trainer.

    Reporting callbacks are normally driven by `Trainer`. Standalone validation has
    no Trainer, so start the callbacks on the first log, forward the metrics, and
    close them at the end. Set WANDB_RUN_ID and WANDB_RESUME=allow in the eval job's
    environment to attach to the training run's dashboard.
    """

    def __init__(self, args, processing_class=None):
        from transformers.integrations import get_reporting_integration_callbacks
        from transformers.trainer_callback import TrainerState

        self.args = args
        self.processing_class = processing_class
        self.state = TrainerState()
        self.callbacks = [
            cls() for cls in get_reporting_integration_callbacks(args.report_to)
        ]
        self.model = None
        self.started = False

    def log(self, metrics: dict, step: int, model=None):
        if not self.callbacks:
            return
        if model is not None:
            self.model = model
        self.state.global_step = step
        if not self.started:
            for callback in self.callbacks:
                callback.on_train_begin(
                    self.args,
                    self.state,
                    None,
                    model=self.model,
                    processing_class=self.processing_class,
                )
            self.started = True
        for callback in self.callbacks:
            callback.on_log(
                self.args,
                self.state,
                None,
                model=self.model,
                processing_class=self.processing_class,
                logs=dict(metrics),
            )

    def close(self):
        if not self.started:
            return
        for callback in self.callbacks:
            callback.on_train_end(
                self.args,
                self.state,
                None,
                model=self.model,
                processing_class=self.processing_class,
            )


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
    tokenizer.padding_side = "right" if data_args.padding_side == "right" else "left"

    torch_dtype = (
        torch.bfloat16
        if training_args.bf16
        else torch.float16
        if training_args.fp16
        else torch.float32
    )
    reporter = _Reporter(training_args, tokenizer)

    try:
        for checkpoint in checkpoints:
            name = os.path.basename(os.path.normpath(checkpoint))
            marker = (
                Path(validate_args.validate_output_dir)
                / name
                / "aggregate.metrics.json"
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
            reporter.log(flat, _step_of(checkpoint), model)
            logger.info(
                "[validate] %s mean NDCG@10=%.4f",
                name,
                flat.get("eval/mean/ndcg@10", float("nan")),
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        reporter.close()


if __name__ == "__main__":
    main()
