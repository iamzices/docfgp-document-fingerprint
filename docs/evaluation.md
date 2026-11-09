# Evaluation

This report covers two completed checks: cross-format retrieval on the office2pdf business golden corpus and validation of the published RVL-CDIP duplicate metadata. The retrieval figures below come from the Office corpus.

## Office source-to-PDF retrieval

The [business golden corpus](https://github.com/developer0hye/office2pdf/blob/main/tests/golden_mocks/business/README.md) pairs 10 DOCX and 10 PPTX files with PDF exports from Microsoft Office. The evaluator fingerprinted the 20 Office sources once, indexed their fingerprint evidence, then fingerprinted and queried the 20 PDF exports once each. The 20 manifest pairs were positives; the other 380 source/PDF pairs were treated as negatives.

The run took place on **9 November 2026** on Windows with Python 3.14.7, office2pdf 0.10.0, Tesseract 5.5.3, and an AMD Ryzen 5 7500F.

| Measure | Result |
| --- | ---: |
| Historical candidates / queries | 20 / 20 |
| Positive / negative pairs | 20 / 380 |
| Precision | 1.000 |
| Recall | 1.000 |
| Retrieval recall | 1.000 |
| Recall@10 | 1.000 |
| False-positive rate | 0.000 |
| CPU / wall time | 15.08 / 14.09 seconds |
| Sampled peak RAM | 150.24 MB |

The corpus is small and synthetic. Office observations informed the final native-text provenance rule, so these numbers describe performance on a development corpus. Accuracy on other collections requires independent evaluation.

## RVL-CDIP metadata validation

The evaluator joined the `documents` and `pairs` configurations of [RVL-CDIP Duplicates](https://huggingface.co/datasets/stefan-hf/rvlcdip-duplicates) at revision `457749089e2b729d8b82f64166a379a0c6c4e21e`. The check validated query IDs, original image-path agreement, and consistency between each query's duplicate flag and its pair judgments.

| Split | Queries with judgments | Positive pairs | Other judged pairs | Excluded query without judgments |
| --- | ---: | ---: | ---: | ---: |
| Validation | 40,000 | 69,172 | 330,828 | 0 |
| Test | 39,999 | 68,290 | 331,700 | 1 |

This check measured label and path consistency. The dataset's pairs cover ten same-class candidates per query selected by another system, and its duplicate labels are heuristic. The evaluator scores precision and false-positive rate on judged pairs and reports predictions on unjudged pairs separately. The accuracy figures above come from the Office retrieval run.

## Retrieval and scoring protocol

The corpus evaluators share a sequential protocol. They fingerprint each historical candidate once and retain fingerprint-derived evidence in a temporary disk-backed index. Each query is fingerprinted once. Exact hashes, MinHash bands, and visual keys retrieve candidates from that index. Candidate indexing and ranking use fingerprints; corpus labels enter after ranking to score the results. The temporary index is deleted after the run.

For a retrieved pair, the text score is the larger of word and character MinHash agreement. Equal raw hashes or equal nonempty canonical hashes score 1. The visual score greedily assigns distinct page-hash pairs and divides their summed similarity by the larger page count. When every page on both sides has usable native text and OCR was not attempted, the text score takes precedence over visual layout. Otherwise the larger score is used. A score of at least **0.8** is counted as a predicted match. This threshold is an evaluator policy. These corpora measure document equivalence, so the evaluators omit shingle occurrence counts; applications can retain counts for directional overlap calculations.

Precision is $TP/(TP+FP)$, recall is $TP/(TP+FN)$, and false-positive rate is $FP/(FP+TN)$. Retrieval recall counts labeled positives reached by any lookup key before score filtering. Recall@K counts labeled positives in the first K ranked candidates, divided by all labeled positives. The report includes raw pair counts to make each denominator visible.

CPU time combines Python process time with sampled child-process time. Peak RAM is the maximum sampled resident memory of the process tree at 10 ms intervals; short-lived peaks can be missed. A document-processing failure stops a run instead of yielding a complete-looking report.

## Reproducing the Office run

Install the library with its development dependencies, obtain the business golden corpus and the office2pdf and Tesseract executables, then run from the repository root:

```bash
poetry install
poetry run python -m scripts.evaluate_office /path/to/business --office2pdf /path/to/office2pdf --tesseract /path/to/tesseract
```

The command prints the metrics and counts as JSON. Record the corpus revision, command, hardware, operating system, Python version, and converter/OCR versions with any subsequent result. Corpus evaluations are separate from CI.
