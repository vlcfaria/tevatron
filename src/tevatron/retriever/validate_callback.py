"""Trainer callback that triggers checkpoint validation on save."""

import logging
import os
import subprocess

import torch.distributed as dist
from transformers import TrainerCallback

from tevatron.retriever.arguments import DataArguments, ValidateArguments
from tevatron.retriever.checkpoint_validation import (
    CHECKPOINT_PREFIX,
    evaluate_model,
)

logger = logging.getLogger(__name__)


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
            if args.should_save and state.is_world_process_zero:
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

        if model is None and self.trainer is not None:
            model = self.trainer.model
        if model is None:
            raise RuntimeError("sync validation requires the trainer's model")

        was_training = model.training
        should_eval = args.should_save and state.is_world_process_zero
        with self._gather_if_zero3(model):
            if should_eval:
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
