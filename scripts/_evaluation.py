"""Shared indexed-retrieval baseline for independently maintained benchmark adapters."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import psutil

from docfgp import DocumentFingerprint

THRESHOLD = 0.8


def _features(fingerprint: DocumentFingerprint) -> dict[str, Any]:
    """Keep only fingerprint-derived evidence needed for ranking."""
    return {
        "raw": fingerprint.exact_hashes["raw_sha256"],
        "canonical": fingerprint.exact_hashes["canonical_sha256"],
        "word": fingerprint.minhash,
        "char": fingerprint.char_minhash,
        "pages": [page.phash for page in fingerprint.page_hashes],
        "page_count": fingerprint.page_count,
        "native_text": bool(fingerprint.quality.pages)
        and all(
            page.source == "native" and page.text_chars > 0 and not page.ocr_attempted
            for page in fingerprint.quality.pages
        ),
    }


def _keys(fingerprint: DocumentFingerprint) -> set[str]:
    """Add exact-hash lookup to the library's approximate lookup keys."""
    keys = {
        key for name in ("word_lsh", "char_lsh", "visual") for key in fingerprint.lookup_keys[name]
    }
    for name, value in fingerprint.exact_hashes.items():
        if value:
            keys.add(f"exact:{name}:{value}")
    return keys


def _score(query: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    """Rank an indexed pair using the strongest available feature evidence."""
    if query["raw"] == candidate["raw"] or (
        query["canonical"] and query["canonical"] == candidate["canonical"]
    ):
        return 1.0

    def minhash(name: str) -> float:
        left, right = query[name], candidate[name]
        return (
            sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
            if left and right
            else 0.0
        )

    visual = 0.0
    if query["pages"] and candidate["pages"]:
        query_hashes = [int(page, 16) for page in query["pages"]]
        candidate_hashes = [int(page, 16) for page in candidate["pages"]]
        similarities = sorted(
            (
                1 - (page ^ other).bit_count() / 64,
                left_index,
                right_index,
            )
            for left_index, page in enumerate(query_hashes)
            for right_index, other in enumerate(candidate_hashes)
        )
        used_query: set[int] = set()
        used_candidate: set[int] = set()
        shared = 0.0
        for similarity, left_index, right_index in reversed(similarities):
            if left_index not in used_query and right_index not in used_candidate:
                used_query.add(left_index)
                used_candidate.add(right_index)
                shared += similarity
        visual = shared / max(query["page_count"], candidate["page_count"])
    text = max(minhash("word"), minhash("char"))
    # Conflicting native text is stronger evidence than similar page layout.
    if (
        query["native_text"]
        and candidate["native_text"]
        and query["canonical"]
        and candidate["canonical"]
    ):
        return text
    return max(text, visual)


class _Index:
    """Store evaluator-owned fingerprints and lookup postings on temporary disk."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA cache_size=-32768")
        self.connection.execute(
            "CREATE TABLE documents (id TEXT PRIMARY KEY, evidence TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE postings (key TEXT NOT NULL, id TEXT NOT NULL, "
            "PRIMARY KEY (key, id)) WITHOUT ROWID"
        )
        self.count = 0

    def add(self, identity: str, fingerprint: DocumentFingerprint) -> None:
        """Insert one processed document and its lookup postings."""
        self.connection.execute(
            "INSERT INTO documents VALUES (?, ?)",
            (identity, json.dumps(_features(fingerprint), separators=(",", ":"))),
        )
        self.connection.executemany(
            "INSERT INTO postings VALUES (?, ?)",
            ((key, identity) for key in sorted(_keys(fingerprint))),
        )
        self.count += 1
        if self.count % 1000 == 0:
            self.connection.commit()

    def lookup(self, fingerprint: DocumentFingerprint) -> set[str]:
        """Retrieve candidates using indexed keys, without scanning history."""
        found: set[str] = set()
        keys = sorted(_keys(fingerprint))
        for start in range(0, len(keys), 500):
            batch = keys[start : start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT id FROM postings WHERE key IN ({placeholders})", batch
            )
            found.update(row[0] for row in rows)
        return found

    def evidence(self, identity: str) -> dict[str, Any]:
        """Load a retained fingerprint summary for one retrieved candidate."""
        row = self.connection.execute(
            "SELECT evidence FROM documents WHERE id=?", (identity,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Candidate is absent from index: {identity}")
        return json.loads(row[0])

    def contains(self, identity: str) -> bool:
        """Check whether a judged candidate was actually ingested."""
        return (
            self.connection.execute("SELECT 1 FROM documents WHERE id=?", (identity,)).fetchone()
            is not None
        )

    def close(self) -> None:
        """Close the benchmark's temporary database."""
        self.connection.close()


def evaluate(
    candidates: Iterable[tuple[str, Path]],
    queries: Iterable[tuple[str, Path]],
    positives: Mapping[str, set[str]],
    fingerprint: Callable[[Path], DocumentFingerprint],
    *,
    negatives: Mapping[str, set[str]] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Evaluate sequential ingestion against labeled document-level pairs.

    Args:
        candidates: Historical document IDs and paths, each processed once.
        queries: Incoming document IDs and paths, each processed once.
        positives: Known positive candidate IDs for each query.
        fingerprint: Fingerprint processor for this corpus.
        negatives: Known negatives; omitted when all unlabeled pairs are negative.
        top_k: Rank cutoff for recall at K.

    Returns:
        Retrieval, classification, and sampled resource measurements.

    Raises:
        ValueError: Labels or document IDs do not agree with ingestion.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    process = psutil.Process(os.getpid())
    peak = process.memory_info().rss
    child_cpu: dict[tuple[int, float], float] = {}
    finished = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not finished.is_set():
            resident = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    resident += child.memory_info().rss
                    cpu = child.cpu_times()
                    identity = child.pid, child.create_time()
                    child_cpu[identity] = max(child_cpu.get(identity, 0.0), cpu.user + cpu.system)
                except psutil.Error:
                    continue
            peak = max(peak, resident)
            finished.wait(0.01)

    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    counts = dict.fromkeys(
        (
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
            "at_k",
            "retrieved_positive",
            "unjudged_predictions",
        ),
        0,
    )
    query_count = 0
    positive_count = 0
    judged_negative_count = 0
    missed: list[str] = []
    false_positives: list[dict[str, str | float]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="docfgp-evaluation-") as directory:
            index = _Index(Path(directory) / "index.sqlite")
            try:
                for identity, path in candidates:
                    index.add(identity, fingerprint(path))
                index.connection.commit()
                candidate_count = index.count
                judged_candidates = {
                    candidate for group in positives.values() for candidate in group
                }
                if negatives:
                    judged_candidates.update(
                        candidate for group in negatives.values() for candidate in group
                    )
                absent = sorted(
                    candidate for candidate in judged_candidates if not index.contains(candidate)
                )
                if absent:
                    raise ValueError(f"Judged candidates were not ingested: {absent[:3]}")
                seen_queries: set[str] = set()
                for identity, path in queries:
                    if identity in seen_queries:
                        raise ValueError(f"Duplicate query ID: {identity}")
                    seen_queries.add(identity)
                    known_positive = positives.get(identity, set())
                    known_negative = (
                        negatives.get(identity, set()) if negatives is not None else set()
                    )
                    if known_positive & known_negative:
                        raise ValueError(f"Conflicting judgments for {identity}")
                    query = fingerprint(path)
                    query_features = _features(query)
                    found = index.lookup(query)
                    ranked = sorted(
                        (
                            (
                                candidate,
                                _score(query_features, index.evidence(candidate)),
                            )
                            for candidate in found
                        ),
                        key=lambda pair: (-pair[1], pair[0]),
                    )
                    predicted = {candidate for candidate, score in ranked if score >= THRESHOLD}
                    counts["true_positive"] += len(predicted & known_positive)
                    counts["false_negative"] += len(known_positive - predicted)
                    counts["retrieved_positive"] += len(found & known_positive)
                    counts["at_k"] += len(known_positive & {item for item, _ in ranked[:top_k]})
                    if negatives is None:
                        counts["false_positive"] += len(predicted - known_positive)
                        counts["true_negative"] += candidate_count - len(predicted | known_positive)
                        judged_negative_count += candidate_count - len(known_positive)
                        false_ids = predicted - known_positive
                    else:
                        counts["false_positive"] += len(predicted & known_negative)
                        counts["true_negative"] += len(known_negative - predicted)
                        counts["unjudged_predictions"] += len(
                            predicted - known_positive - known_negative
                        )
                        judged_negative_count += len(known_negative)
                        false_ids = predicted & known_negative
                    for candidate, score in ranked:
                        if candidate in false_ids and len(false_positives) < 20:
                            false_positives.append(
                                {"query": identity, "candidate": candidate, "score": score}
                            )
                    positive_count += len(known_positive)
                    query_count += 1
                    if known_positive - predicted and len(missed) < 20:
                        missed.append(identity)
                if set(positives) - seen_queries:
                    raise ValueError(
                        "Positive labels contain queries absent from the evaluated split"
                    )
                if negatives is not None and set(negatives) - seen_queries:
                    raise ValueError(
                        "Negative labels contain queries absent from the evaluated split"
                    )
            finally:
                index.close()
    finally:
        finished.set()
        worker.join(timeout=1)
    tp = counts["true_positive"]
    fp = counts["false_positive"]
    fn = counts["false_negative"]
    tn = counts["true_negative"]
    return {
        "candidate_count": candidate_count,
        "query_count": query_count,
        "positive_pairs": positive_count,
        "judged_negative_pairs": judged_negative_count,
        "policy": (
            "indexed exact/LSH/visual retrieval; "
            f"threshold={THRESHOLD}; native text resolves conflicting visual evidence"
        ),
        "top_k": top_k,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "retrieval_recall": counts["retrieved_positive"] / positive_count
        if positive_count
        else 0.0,
        "recall_at_k": counts["at_k"] / positive_count if positive_count else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        **counts,
        "cpu_seconds": time.process_time() - start_cpu + sum(child_cpu.values()),
        "wall_seconds": time.perf_counter() - start_wall,
        "peak_ram_bytes": peak,
        "resource_sample_interval_seconds": 0.01,
        "missed_query_examples": missed,
        "false_positive_examples": false_positives,
    }
