"""Sample validation (dev set) queries from a tevatron-like training dataset. Useful for DPR-like evaluation.

A Tevatron training dataset is defined as:

    {
        "query_id": "<query id>",
        "query": "<query text>",
        "positive_passages": [
            {"docid": "<passage id>", "title": "<passage title>", "text": "<passage body>"}
        ],
        "negative_passages": [
            {"docid": "<passage id>", "title": "<passage title>", "text": "<passage body>"}
        ]
    }

This script allows extracting a portion of a Tevatron in two ways: random or balanced.
On a balanced strategy, different subsets of the training data is split equally. There are multiple ways 
a HF dataset can be separated: splits, subsets and columns (like rlhn/rlhn-680K). This scripts follows the RLHN
approach and allows grouping by only by column

Usage:
    python -m tevatron.utils.sample_dev_queries \
        --dataset_name rlhn/rlhn-680K --dataset_split train \
        --group_by column --group_column subset \
        --num_queries 1000 --seed 1234 \
        --output_dir data/dev/rlhn_7x1k
"""

import argparse
import os

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample a validation set from a tevatron training dataset."
    )

    p.add_argument("--dataset_name", help="HF dataset id e.g. rlhn/rlhn-680K")

    p.add_argument(
        "--dataset_config",
        help="HF dataset config, useful for datasets with sub-datasets",
    )

    p.add_argument("--dataset_path", help="Path to local data files or directory")

    p.add_argument("--dataset_split", default="train", help="Dataset split")

    p.add_argument("--seed", default=42, type=int, help="Seed to pick dev queries")

    p.add_argument(
        "--num_queries",
        required=True,
        type=float,
        help="Number of dev queries to sample per group if applicable. Passing a float picks a % of each group.",
    )

    p.add_argument(
        "--output_dir",
        required=True,
        help="Output directory of dataset split preparation",
    )

    p.add_argument("--group_by", choices=["column"], help="Value to group by columns")

    p.add_argument("--group_column", help="Column to group by on stratified")

    args = p.parse_args()

    if not args.dataset_name and not args.dataset_path:
        p.error(
            "Dataset name and dataset path were not informed. Please inform a dataset"
        )
    if args.group_by and not args.group_column:
        p.error(
            "No group column was informed. Please inform a group column with --group_column"
        )

    return args


def balance_split_dataset(
    dataset: Dataset, group_column: str, num_queries: float, seed: int
) -> dict[str, DatasetDict]:
    mapper = dataset.features[group_column].int2str
    unique_labels = dataset.unique(group_column)

    subsets = {}

    for label in unique_labels:
        class_subset = dataset.filter(lambda x: x[group_column] == label)

        if not num_queries.is_integer():
            label_num_queries = num_queries * len(class_subset)
        else:
            label_num_queries = num_queries

        label_num_queries = int(label_num_queries)

        if len(class_subset) < label_num_queries:
            raise ValueError(
                f"Label {label} only has {len(class_subset)} instances, which is less than requested {num_queries}."
            )

        class_split = class_subset.train_test_split(
            test_size=label_num_queries, seed=seed
        )

        subsets[mapper(label)] = class_split

    return subsets


def save_test_query_dataset(dataset: DatasetDict, path: str):
    dev_dict = DatasetDict({"test": dataset["test"]})
    dev_dict.save_to_disk(path)


def save_train_query_dataset(dataset: Dataset, path: str):
    train_dict = DatasetDict({"train": dataset})
    train_dict.save_to_disk(path)


if __name__ == "__main__":
    args = parse_args()

    dataset = load_dataset(
        args.dataset_name or args.dataset_path, name=args.dataset_config
    )

    os.makedirs(args.output_dir, exist_ok=True)
    dev_subset_dir = os.path.join(args.output_dir, "dev/")
    train_subset_dir = os.path.join(args.output_dir, "train/")

    if args.group_by:
        dataset = dataset.class_encode_column(args.group_column)
        separated = balance_split_dataset(
            dataset[args.dataset_split], args.group_column, args.num_queries, args.seed
        )

        # Save one dataset per group
        for subset_name, dataset_dict in separated.items():
            group_name = os.path.join(dev_subset_dir, subset_name)
            save_test_query_dataset(dataset_dict, group_name)

            print(f"Saved '{subset_name}'")

        # Save a single train set
        save_train_query_dataset(
            concatenate_datasets([v["train"] for v in separated.values()]),
            train_subset_dir,
        )

    else:
        num_queries = int(
            args.num_queries
            if args.num_queries.is_integer()
            else len(dataset[args.dataset_split]) * args.num_queries
        )
        separated = dataset[args.dataset_split].train_test_split(
            test_size=num_queries, seed=args.seed
        )

        save_test_query_dataset(separated, dev_subset_dir)
        save_train_query_dataset(separated["train"], train_subset_dir)
