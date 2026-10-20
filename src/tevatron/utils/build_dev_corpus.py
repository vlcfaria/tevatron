"""Build a DPR-like dev corpus for a small validation set. 

The dev corpus is built by indexing the full corpus and sampling the top-k passages from each dev query.
The passages are then deduplicated, and the positive + hard negatives are also injected into the corpus, building a
smaller corpus that serves as a proxy for an evaluation set. Such method was proposed by DPR and emphasized with Asyncval

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

Usage:
    python -m python -m tevatron.utils.build_dev_corpus --model_name_or_path=intfloat/e5-base-unsupervised \
        --query_max_len=512 --passage_max_length=512 --per_device_eval_batch_size=512 --attn_implementation sdpa \
        --bf16=false --fp16=true --pooling=mean --normalize=true  '--query_prefix=query: ' '--passage_prefix=passage: ' \
        --depth=1000 --batch_size=64 --shard_search --gpu_float32  --dataset_map=./map.json --encoding_cache_dir=./dev-encoding \
        --output_dir="./dev-corpus-final"
"""

import json
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


@dataclass
class DatasetInfo:
    name: str
    encode_corpus_config: str
    encode_corpus_name: str
    query_corpus_config: str
    query_corpus_name: str


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

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    print(f"[Process {rank}] Initialized on device: {device}")

    data_args = (
        output.get_query_data_args(data_args)
        if is_query
        else output.get_corpus_data_args(data_args, rank, num_shards)
    )

    if os.path.exists(data_args.encode_output_path):
        print(f"Skipping encoding {data_args.encode_output_path}: File already exists.")
        return

    print(data_args)

    main_encode(model_args, data_args, training_args)


def encode_dataset(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    output: DatasetOutput,
):
    num_proc = torch.cuda.device_count()

    try:
        print("Encoding Corpus")
        mp.spawn(
            fn=encode_worker,
            args=(num_proc, model_args, data_args, training_args, output, False),
            nprocs=num_proc,
            join=True,
        )

        print("Encoding Queries")
        mp.spawn(
            fn=encode_worker,
            args=(num_proc, model_args, data_args, training_args, output, True),
            nprocs=num_proc,
            join=True,
        )

    except Exception as e:
        print(f"Fatal error ocurred in a subprocess: \n{e}")


def search_dataset(
    search_args: Namespace,
    output: DatasetOutput,
):
    output_dir = output.get_rank_output(search_args.save_text)

    if os.path.exists(output_dir):
        print(f"Skipping search: {output_dir} already exists")
        return output_dir

    search_args.passage_reps = str(output.get_passage_reps())
    search_args.query_reps = str(output.get_query_file_name())
    search_args.save_ranking_to = output_dir
    main_search(search_args)

    return output_dir


def build_dev_corpus(
    dataset: DatasetOutput, docs: np.ndarray, data_args: DataArguments
):
    # Get corpus for fetching actual text
    corpus = EncodeDataset(dataset.get_corpus_data_args(data_args, 0, 1))
    docs_set = set(docs)

    # Get dev data for neg/pos training passages
    dev_data = TrainDataset(dataset.get_query_data_args(data_args))
    dev_passages = {
        text["title"] + " " + text["text"]
        for row in dev_data.train_data.select_columns(
            ["positive_passages", "negative_passages"]
        )
        for text in row["positive_passages"] + row["negative_passages"]
    }

    filtered_corpus = corpus.encode_data.filter(
        lambda x: (
            x["docid"] in docs_set
            and x["text"]
            and x["text"] not in dev_passages
            and x["title"] + " " + x["text"]
        )
    )

    # Stitch our own documents
    dev_passages = list(dev_passages)
    passage_corpus = {
        "docid": [f"assigned-{i}" for i in range(len(dev_passages))],
        "title": [""] * len(dev_passages),
        "text": dev_passages,
    }

    augmented_corpus = Dataset.from_dict(passage_corpus)
    dev_corpus = concatenate_datasets([filtered_corpus, augmented_corpus])

    print(
        f"Picked {len(dev_corpus)} documents, with {len(filtered_corpus)} original documents and {len(augmented_corpus)} augmented documents"
    )

    return dev_corpus


def main(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    search_args: Namespace,
    datasets: list[DatasetInfo],
    encoding_cache_dir: str,
):
    num_gpus = torch.cuda.device_count()
    os.makedirs(training_args.output_dir, exist_ok=True)

    print(f"Encoding with {num_gpus} GPUs")

    for d in datasets:
        output = DatasetOutput(d, encoding_cache_dir)

        print(f"Encoding {d.name}")
        encode_dataset(
            model_args,
            data_args,
            training_args,
            output=output,
        )

        print(f"Searching {d.name}")
        ranking_file = search_dataset(search_args, output)

        with open(ranking_file, "rb") as f:
            # docs -> (num_queries x depth)
            _, docs = pickle.load(f)

        dev_corpus = build_dev_corpus(output, np.unique(docs.flatten()), data_args)
        dev_corpus.save_to_disk(Path(training_args.output_dir) / d.name)


def parse_dev_corpus_args():
    p = ArgumentParser()

    p.add_argument("--dataset_map", type=str, required=True)
    p.add_argument("--encoding_cache_dir", type=str, required=True)

    args, _ = p.parse_known_args()

    with open(args.dataset_map, "r") as f:
        mapper = json.load(f)

    return [DatasetInfo(**val) for val in mapper], args


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
    )
