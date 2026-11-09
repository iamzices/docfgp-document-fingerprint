"""Evaluate RVL-CDIP duplicate retrieval against published test/train pair judgments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from docfgp import FingerprintConfig, Fingerprinter
from scripts._evaluation import evaluate


def _relative_path(value: str) -> str:
    """Validate an original RVL-CDIP relative image path."""
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts or path.suffix.lower() != ".tif":
        raise ValueError(f"Invalid RVL-CDIP image path: {value}")
    return path.as_posix()


def _training_images(label_file: Path, images_root: Path) -> Iterable[tuple[str, Path]]:
    """Yield the full original training split without using duplicate labels."""
    with label_file.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"Invalid training label at line {line_number}")
            identity = _relative_path(fields[0])
            yield identity, images_root / identity


def _judgments(
    documents: Iterable[dict[str, Any]], pairs: Iterable[dict[str, Any]], images_root: Path
) -> tuple[list[tuple[str, Path]], dict[str, set[str]], dict[str, set[str]], int]:
    """Join query paths to published judged pairs and validate label consistency."""
    queries: list[tuple[str, Path]] = []
    query_paths: dict[str, str] = {}
    expected: dict[str, bool] = {}
    excluded = 0
    for row in documents:
        if row["has_train_duplicate"] is None:
            excluded += 1
            continue
        identity = str(row["id"])
        if identity in query_paths:
            raise ValueError(f"Duplicate query ID in RVL metadata: {identity}")
        path = _relative_path(str(row["rvlcdip_path"]))
        query_paths[identity] = path
        expected[identity] = bool(row["has_train_duplicate"])
        queries.append((identity, images_root / path))
    positives: dict[str, set[str]] = defaultdict(set)
    negatives: dict[str, set[str]] = defaultdict(set)
    for row in pairs:
        identity = str(row["id"])
        if identity not in query_paths:
            raise ValueError(f"Pair refers to a query absent from documents: {identity}")
        if _relative_path(str(row["rvlcdip_path"])) != query_paths[identity]:
            raise ValueError(f"Query path disagrees between RVL configurations: {identity}")
        candidate = _relative_path(str(row["train_rvlcdip_path"]))
        target = positives if row["duplicate"] else negatives
        target[identity].add(candidate)
    for identity, has_duplicate in expected.items():
        if has_duplicate != bool(positives.get(identity)):
            raise ValueError(f"Duplicate flags disagree with pair judgments: {identity}")
        if positives[identity] & negatives[identity]:
            raise ValueError(f"Conflicting pair judgments: {identity}")
    return queries, positives, negatives, excluded


def main(argv: list[str] | None = None) -> int:
    """Run the RVL-CDIP indexed-retrieval baseline.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images_root", type=Path, help="Root of the original RVL-CDIP images")
    parser.add_argument("train_labels", type=Path, help="Original RVL-CDIP train.txt label file")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--dataset", default="stefan-hf/rvlcdip-duplicates")
    parser.add_argument("--revision", help="Optional pinned Hugging Face dataset revision")
    parser.add_argument("--tesseract", help="Tesseract executable for scanned-image text")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install the benchmark extra: poetry install -E benchmark") from exc
    revision = args.revision
    if revision is None and not Path(args.dataset).exists():
        revision = HfApi().dataset_info(args.dataset).sha
    options = {"revision": revision} if revision else {}
    documents = load_dataset(args.dataset, "documents", split=args.split, **options)
    pairs = load_dataset(args.dataset, "pairs", split=args.split, **options)
    queries, positives, negatives, excluded = _judgments(documents, pairs, args.images_root)
    fingerprinter = Fingerprinter(
        FingerprintConfig(tesseract_path=args.tesseract, include_shingle_counts=False)
    )
    report = evaluate(
        _training_images(args.train_labels, args.images_root),
        queries,
        positives,
        fingerprinter.fingerprint,
        negatives=negatives,
        top_k=args.top_k,
    )
    report.update(
        corpus="RVL-CDIP Duplicates",
        split=args.split,
        dataset=args.dataset,
        revision=revision,
        excluded_unjudged_queries=excluded,
        annotation_scope=(
            "published judged pairs only; unknown pairs excluded from precision and FPR"
        ),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
