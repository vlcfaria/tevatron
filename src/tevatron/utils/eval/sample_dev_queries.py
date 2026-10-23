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
import logging
import os
import random
from collections import defaultdict

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)


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
        help="Number of unique dev queries to sample per group if applicable. Passing a float picks a fraction of each group's unique queries.",
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


def split_by_query_id(
    dataset: Dataset, num_queries: float, seed: int, group_column: str | None = None
) -> DatasetDict:
    """Split rows so every query id belongs entirely to train or test.
    This covers the case where duplicate query ids are present, and they contain the same textual query
    (RLHN case)

    If `group_column` is given, the split unit is `(group_column, query_id)`, so
    the same query can still be split independently across different groups.
    """

    if group_column is not None:
        units = list(zip(dataset[group_column], dataset["query_id"]))
    else:
        units = list(dataset["query_id"])

    unit_to_indices = defaultdict(list)
    for index, unit in enumerate(units):
        unit_to_indices[unit].append(index)

    unique_units = list(unit_to_indices)
    if isinstance(num_queries, float) and not num_queries.is_integer():
        count = int(num_queries * len(unique_units))
    else:
        count = int(num_queries)

    if count < 1:
        raise ValueError("num_queries must select at least one query")
    if count > len(unique_units):
        raise ValueError(
            f"Requested {count} queries but only found {len(unique_units)} unique queries"
        )

    # Split train + dev
    dev_units = set(random.Random(seed).sample(unique_units, count))
    dev_indices = []
    train_indices = []
    for unit, indices in unit_to_indices.items():
        target = dev_indices if unit in dev_units else train_indices
        target.extend(indices)

    dev_indices.sort()
    train_indices.sort()
    return DatasetDict(
        {
            "train": dataset.select(train_indices),
            "test": dataset.select(dev_indices),
        }
    )


def _merge_passages(rows: list[dict], field: str) -> list[dict]:
    seen = set()
    merged = []
    for row in rows:
        for passage in row.get(field, []):
            key = passage.get("docid")
            if key is None:
                key = (passage.get("title"), passage.get("text"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(passage)
    return merged


def merge_query_rows(dataset: Dataset, group_column: str | None = None) -> Dataset:
    """Merge rows with the same `(group, query_id)` into one dev query row."""

    groups = {}
    order = []
    for row in dataset:
        key = (
            (row[group_column], row["query_id"])
            if group_column is not None
            else row["query_id"]
        )
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    merged_rows = []
    for key in order:
        rows = groups[key]
        base = dict(rows[0])
        if any(row["query"] != base["query"] for row in rows[1:]):
            raise ValueError(f"query_id {key!r} has multiple query texts")
        base["positive_passages"] = _merge_passages(rows, "positive_passages")
        base["negative_passages"] = _merge_passages(rows, "negative_passages")
        merged_rows.append(base)

    return Dataset.from_list(merged_rows, features=dataset.features)


def balance_split_dataset(
    dataset: Dataset, group_column: str, num_queries: float, seed: int
) -> dict[str, DatasetDict]:
    "Splits a dataset, picking `num_queries` unique queries per subset."

    mapper = dataset.features[group_column].int2str
    unique_labels = dataset.unique(group_column)
    subsets = {}

    for label in unique_labels:
        class_subset = dataset.filter(lambda x: x[group_column] == label)
        subsets[mapper(label)] = split_by_query_id(
            class_subset, num_queries, seed, group_column=group_column
        )

    return subsets


def save_test_query_dataset(
    dataset: DatasetDict, path: str, group_column: str | None = None
):
    dev_dict = DatasetDict(
        {"test": merge_query_rows(dataset["test"], group_column=group_column)}
    )
    dev_dict.save_to_disk(path)


def save_train_query_dataset(dataset: Dataset, path: str):
    train_dict = DatasetDict({"train": dataset})
    train_dict.save_to_disk(path)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
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
            save_test_query_dataset(
                dataset_dict, group_name, group_column=args.group_column
            )

            logger.info("Saved '%s'", subset_name)

        # Save a single train set
        save_train_query_dataset(
            concatenate_datasets([v["train"] for v in separated.values()]),
            train_subset_dir,
        )

    else:
        separated = split_by_query_id(
            dataset[args.dataset_split], args.num_queries, args.seed
        )

        save_test_query_dataset(separated, dev_subset_dir)
        save_train_query_dataset(separated["train"], train_subset_dir)
