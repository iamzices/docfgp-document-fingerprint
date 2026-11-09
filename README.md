# DocFGP

[![PyPI version](https://img.shields.io/pypi/v/docfgp)](https://pypi.org/project/docfgp/)
[![PyPI downloads](https://img.shields.io/pepy/dt/docfgp)](https://pepy.tech/projects/docfgp)
[![CI](https://github.com/iamzices/docfgp-document-fingerprint/actions/workflows/ci.yml/badge.svg)](https://github.com/iamzices/docfgp-document-fingerprint/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/github/license/iamzices/docfgp-document-fingerprint)](LICENSE)

DocFGP creates text and visual fingerprints for finding duplicates, near-duplicates, and shared passages across document formats.

## Features

- PDF, DOCX, PPTX, PNG, JPEG, TIFF, BMP, and WebP support
- Exact, near-duplicate, and partial-match
- OCR and visual fingerprints for scanned documents

## Requirements

- Python 3.11+
- [office2pdf](https://github.com/developer0hye/office2pdf)
- [Tesseract 5](https://github.com/tesseract-ocr/tesseract)

## Installation

```bash
pip install docfgp
```

## Python API

```python
from docfgp import Fingerprinter

fingerprint = Fingerprinter().fingerprint("report.pdf")
record = fingerprint.to_dict()
```

| Field | Description |
| --- | --- |
| `schema_version` | Fingerprint data format version |
| `algorithm_version` | Feature-generation version |
| `source_format` | Detected input format |
| `byte_size` | Original file size in bytes |
| `page_count` | Number of pages or image frames |
| `exact_hashes` | Raw and canonical-text SHA-256 hashes |
| `lookup_keys` | Word, character, passage, and visual candidate keys |
| `minhash` | Word-shingle MinHash sketch |
| `char_minhash` | Character-shingle MinHash sketch |
| `shingle_counts` | Hashed five-word shingle occurrence counts |
| `shingle_total` | Total five-word shingle occurrences |
| `shingle_unique` | Distinct five-word shingle hashes |
| `passage_anchors` | Selected passage fingerprints |
| `page_hashes` | Per-page visual hashes and lookup keys |
| `quality` | Evidence status, page provenance, and warnings |
| `parameters` | Parameters used to construct the fingerprint |

## CLI

```bash
docfgp report.pdf > report.fingerprint.json
```
