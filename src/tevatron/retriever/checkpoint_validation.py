"""Core checkpoint-validation engine: group discovery, encode/search/score, aggregation."""

import json
import logging
import os
from dataclasses import replace
from pathlib import Path

from tevatron.retriever.arguments import DataArguments, ValidateArguments
from tevatron.retriever.driver.encode import encode
from tevatron.retriever.searcher import FaissFlatSearcher

logger = logging.getLogger(__name__)

DEFAULT_DEV_K_VALUES = (1, 5, 10, 20, 100, 200)
CHECKPOINT_PREFIX = "checkpoint"


def check_validate_args(validate_args: ValidateArguments):
    """Fail fast on invalid validation args"""
    if validate_args.validate_mode not in ("off", "sync", "command"):
        raise ValueError(f"invalid validate_mode {validate_args.validate_mode!r}")
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

    # Dev rows can repeat a query_id, assume they contain the same text and collapse them with a dict
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
