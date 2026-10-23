"""Build a DPR-like dev corpus for a small validation set. 

The dev corpus is built by indexing the full corpus and sampling the top-k passages from each dev query.
The passages are then deduplicated, and the positive + hard negatives are also injected into the corpus, building a
smaller corpus that serves as a proxy for an evaluation set. Such method was proposed by DPR and emphasized with Asyncval.
By default, it injects the positive and negative passages into the dev corpus. Pass `--inject_positives_only` to inject only positive 
passages and let the sampled corpus supply the negatives.

This script assumes a held out dev train dataset. You can extract one with tevatron.utils.sample_dev_queries.py.
You also need a .json mapper file that maps each subset of dev queries into their respective corpus dataset/config, as such:

[
    {
        "name": "arguana",
        "encode_corpus_config": "arguana",
        "encode_corpus_name": "tevatron/beir-corpus",
        "query_corpus_config": "arguana",
        "query_corpus_name": "dev-split/dev"
    }
]

Also writes `<output_dir>/<name>.qrels.json` ({query_id: {assigned_docid: rel}}),
matching tevatron.eval.metrics.score's qrels input. Relevance is graded when the
dev passages carry a `score` field, binary otherwise.

Usage:
    python -m python -m tevatron.utils.build_dev_corpus --model_name_or_path=intfloat/e5-base-unsupervised \
        --query_max_len=512 --passage_max_length=512 --per_device_eval_batch_size=512 --attn_implementation sdpa \
        --bf16=false --fp16=true --pooling=mean --normalize=true  '--query_prefix=query: ' '--passage_prefix=passage: ' \
        --depth=1000 --batch_size=64 --shard_search --gpu_float32  --dataset_map=./map.json --encoding_cache_dir=./dev-encoding \
        --output_dir="./dev-corpus-final"
"""

import json
import logging
import os
import pickle
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import Dataset, concatenate_datasets
from transformers import HfArgumentParser

from tevatron.retriever.arguments import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
)
from tevatron.retriever.dataset import EncodeDataset, TrainDataset
from tevatron.retriever.driver.encode import main as main_encode
from tevatron.retriever.driver.search import get_parser
from tevatron.retriever.driver.search import main as main_search
from tevatron.utils.eval.dedup_methods import (
    DedupConfig,
    build_resolver,
    normalize_text,
)

logger = logging.getLogger(__name__)


def configure_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


@dataclass
class DatasetInfo:
    name: str
    encode_corpus_config: str
    encode_corpus_name: str
    query_corpus_config: str
    query_corpus_name: str
    dedup: dict | None = None


class DatasetOutput:
    def __init__(self, dataset: DatasetInfo, output_dir: str):
        dataset_stem = dataset.encode_corpus_name.split("/")[-1]
        self.config_dir = Path(output_dir, dataset_stem, dataset.encode_corpus_config)
        self.dataset = dataset

        os.makedirs(self.config_dir, exist_ok=True)

    def get_corpus_file_name(self, rank) -> Path:
        return self.config_dir / f"corpus-{rank}.pkl"

    def get_passage_reps(self) -> Path:
        return self.config_dir / "corpus-*.pkl"

    def get_corpus_data_args(
        self, data_args: DataArguments, rank: int, num_shards: int
    ) -> DataArguments:
        return replace(
            data_args,
            dataset_name=self.dataset.encode_corpus_name,
            dataset_config=self.dataset.encode_corpus_config,
            dataset_number_of_shards=num_shards,
            dataset_shard_index=rank,
            encode_output_path=self.get_corpus_file_name(rank),
            encode_is_query=False,
        )

    def get_query_file_name(self) -> Path:
        return self.config_dir / "query.pkl"

    def get_query_data_args(self, data_args: DataArguments) -> DataArguments:
        return replace(
            data_args,
            dataset_name=self.dataset.query_corpus_name,
            dataset_config=self.dataset.query_corpus_config,
            dataset_split="test",
            dataset_number_of_shards=1,
            encode_output_path=self.get_query_file_name(),
            encode_is_query=True,
        )

    def get_rank_output(self, save_text: bool = True) -> Path:
        return self.config_dir / f"rank.{'text' if save_text else 'pkl'}"


def encode_worker(
    rank: int,
    num_shards: int,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    output: DatasetOutput,
    is_query: bool,
):
    """Encode the given corpus shard with a single gpu"""

    configure_logging()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    logger.info("[Process %d] Initialized on device: %s", rank, device)

    data_args = (
        output.get_query_data_args(data_args)
        if is_query
        else output.get_corpus_data_args(data_args, rank, num_shards)
    )

    if os.path.exists(data_args.encode_output_path):
        logger.info(
            "Skipping encoding %s: File already exists.",
            data_args.encode_output_path,
        )
        return

    logger.info("%s", data_args)

    main_encode(
        model_args, data_args, training_args, device=torch.device("cuda", rank)
    )


def encode_dataset(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    output: DatasetOutput,
):
    """ "Encode a corpus dataset and a query datasets

    Splits encoding and shards the corpus dataset among `torch.cuda.device_count()` workers
    """
    num_proc = torch.cuda.device_count()

    logger.info("Encoding Corpus")
    mp.spawn(
        fn=encode_worker,
        args=(num_proc, model_args, data_args, training_args, output, False),
        nprocs=num_proc,
        join=True,
    )

    logger.info("Encoding Queries")
    mp.spawn(
        fn=encode_worker,
        args=(1, model_args, data_args, training_args, output, True),
        nprocs=1,
        join=True,
    )


def search_dataset(
    search_args: Namespace,
    output: DatasetOutput,
) -> Path:
    """Searches an encoded dataset. Returns the ranking path"""
    output_dir = output.get_rank_output(search_args.save_text)

    if os.path.exists(output_dir):
        logger.info("Skipping search: %s already exists", output_dir)
        return output_dir

    search_args.passage_reps = str(output.get_passage_reps())
    search_args.query_reps = str(output.get_query_file_name())
    search_args.save_ranking_to = output_dir
    main_search(search_args)

    return output_dir


def passage_key(p: dict) -> str:
    """Normalize passages with `title` (optional) and `text` fields"""
    return normalize_text((p.get("title") or "") + " " + p["text"])


def build_qrels(dev_data: TrainDataset, assigned_map: dict) -> dict:
    """Build eval compatible `qrels: {query_id: {assigned_docid: relevance}}`.

    Relevance is graded when passages carry a `score` field (e.g. distillation
    datasets), binary 1 otherwise. Pytrec treats rel > 0 as relevant for
    recall/map/precision; NDCG uses the grades.
    """
    qrels = {}
    for row in dev_data.train_data:
        # rows may repeat a query_id with different positives (RLHN case)
        # so we merge isntead of overwriting
        rels = qrels.setdefault(str(row["query_id"]), {})
        for p in row["positive_passages"]:
            rel = float(p["score"]) if "score" in p else 1
            pid = assigned_map[passage_key(p)]
            rels[pid] = max(rels.get(pid, rel), rel)
    return {qid: rels for qid, rels in qrels.items() if rels}


def build_dev_corpus(
    dataset: DatasetOutput,
    retrieved_docs: np.ndarray,
    data_args: DataArguments,
    inject_positives_only: bool = False,
):
    """Builds the dev corpus from a tevatron corpus dataset and a tevatron training dataset

    This effectivaly adds positive training examples (and negative passages if `inject_positives_only` is true).
    Since train <-> corpus dataset passages can differ a bit (RLHN x BEIR) case, we apply deduping
    according to the strategies passed in `dataset.dataset.dedup`.
    """
    # Get corpus for fetching actual text in a single shard
    corpus = EncodeDataset(dataset.get_corpus_data_args(data_args, 0, 1))
    retrieved_docs_set = {str(d) for d in retrieved_docs}

    # Get dev data for neg/pos training passages
    dev_data = TrainDataset(dataset.get_query_data_args(data_args))
    dev_texts = sorted(
        {
            passage_key(p)
            for row in dev_data.train_data.select_columns(
                ["positive_passages", "negative_passages"]
            )
            for p in (
                row["positive_passages"]
                if inject_positives_only
                else row["positive_passages"] + row["negative_passages"]
            )
        }
    )
    assigned_map = {text: f"assigned-{i}" for i, text in enumerate(dev_texts)}

    resolver = build_resolver(dev_texts, dataset.dataset.dedup)
    config = resolver.config

    stats = {"corpus": 0, "matched": 0}
    examples = []

    def keep(x: dict) -> bool:
        if str(x["docid"]) not in retrieved_docs_set:
            return False
        stats["corpus"] += 1
        match = resolver.match(passage_key(x))
        if match is None:
            return True
        stats["matched"] += 1
        if len(examples) < 5:
            examples.append((x["docid"], match))
        return False

    # Dedup corpus passages from train set
    filtered_corpus = corpus.encode_data.filter(keep, load_from_cache_file=False)

    # Log stats
    logger.info(
        "[%s] dedup=%s/%s",
        dataset.dataset.name,
        config.method,
        config.action,
    )
    for method in config.methods:
        logger.info(
            "  %s: %d/%d corpus docs dropped, %d/%d dev passages covered",
            method.method,
            resolver.match_counts[method.method],
            stats["corpus"],
            len(resolver.matched_by_method[method.method]),
            len(resolver.dev_texts),
        )

    for docid, match in examples:
        logger.info(
            "  %s <- dev[%d] %s score=%.3f",
            docid,
            match.dev_index,
            match.method,
            match.score,
        )

    corpus_total = stats["corpus"] or 1
    dev_total = len(resolver.dev_texts) or 1
    logger.info(
        "  total: %d/%d corpus docs dropped (%.1f%%), %d/%d dev passages covered (%.1f%%)",
        stats["matched"],
        stats["corpus"],
        stats["matched"] / corpus_total * 100,
        len(resolver.matched),
        len(resolver.dev_texts),
        len(resolver.matched) / dev_total * 100,
    )

    # Stitch documents from train set, RLHN-style
    passage_corpus = {
        "docid": list(assigned_map.values()),
        "title": [""] * len(assigned_map),
        "text": dev_texts,
    }

    augmented_corpus = Dataset.from_dict(passage_corpus)
    dev_corpus = concatenate_datasets([filtered_corpus, augmented_corpus])

    logger.info(
        "Picked %d documents, with %d original documents and %d augmented documents",
        len(dev_corpus),
        len(filtered_corpus),
        len(augmented_corpus),
    )

    return dev_corpus, build_qrels(dev_data, assigned_map)


def main(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    search_args: Namespace,
    datasets: list[DatasetInfo],
    encoding_cache_dir: str,
    inject_positives_only: bool = False,
):
    configure_logging()
    num_gpus = torch.cuda.device_count()
    os.makedirs(training_args.output_dir, exist_ok=True)

    logger.info("Encoding with %d GPUs", num_gpus)

    for d in datasets:
        output = DatasetOutput(d, encoding_cache_dir)

        logger.info("Encoding %s", d.name)
        encode_dataset(
            model_args,
            data_args,
            training_args,
            output=output,
        )

        logger.info("Searching %s", d.name)
        ranking_file = search_dataset(search_args, output)

        # Extract docids from ranking files
        with open(ranking_file, "rb") as f:
            # docs -> (num_queries x depth)
            _, docs = pickle.load(f)

        dev_corpus, qrels = build_dev_corpus(
            output,
            np.unique(docs.flatten()),
            data_args,
            inject_positives_only=inject_positives_only,
        )

        dev_corpus.save_to_disk(Path(training_args.output_dir) / d.name)
        qrels_path = Path(training_args.output_dir) / f"{d.name}.qrels.json"
        with open(qrels_path, "w") as f:
            json.dump(qrels, f, indent=2)

        # Quick sanity check
        corpus_ids = set(dev_corpus["docid"])
        missing = {
            pid for rels in qrels.values() for pid in rels if pid not in corpus_ids
        }
        assert not missing, (
            f"qrels reference docs missing from corpus: {sorted(missing)[:5]}"
        )

        logger.info("Wrote %d qrels to %s", len(qrels), qrels_path)


def parse_dev_corpus_args():
    p = ArgumentParser()

    p.add_argument("--dataset_map", type=str, required=True)
    p.add_argument("--encoding_cache_dir", type=str, required=True)
    p.add_argument(
        "--inject_positives_only",
        action="store_true",
        help="Inject only positive passages into the dev corpus, skippping sampled negatives.",
    )

    args, _ = p.parse_known_args()

    with open(args.dataset_map, "r") as f:
        mapper = json.load(f)

    datasets = [DatasetInfo(**val) for val in mapper]

    # Fast-fail
    for dataset in datasets:
        DedupConfig.from_dict(dataset.dedup)

    return datasets, args


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    search_args, _ = get_parser(False).parse_known_args()
    datasets, args = parse_dev_corpus_args()

    main(
        model_args,
        data_args,
        training_args,
        search_args,
        datasets,
        args.encoding_cache_dir,
        inject_positives_only=args.inject_positives_only,
    )
