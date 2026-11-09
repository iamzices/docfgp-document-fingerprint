# Applications

DocFGP generates fingerprints for document collections. The application stores them, indexes their keys, scores candidate pairs, and makes ingestion decisions.

## Using a fingerprint in a collection

1. Generate a fingerprint with `Fingerprinter().fingerprint(path)`.
2. Inspect `quality.status`, page provenance, and warnings. An `insufficient_evidence` result supports raw-byte identity only.
3. Store the serialized result with its schema and algorithm versions. Index the exact hashes and the relevant entries of `lookup_keys`.
4. Retrieve candidate fingerprints through those indexes. Compare only compatible versions and calculate the evidence needed for the application's decision.

## Questions the evidence can answer

| Question | Relevant evidence | Interpretation |
| --- | --- | --- |
| Are the files byte-identical? | Raw SHA-256 | Equality establishes byte identity. |
| Did extraction produce the same text across formats? | Nonempty canonical SHA-256 and extraction quality | Equality establishes identity of normalized extracted tokens; layout can differ. |
| Are the documents broadly similar? | Word and character MinHash, reached through LSH keys | Sketch agreement estimates set resemblance. |
| Is one document substantially contained in another? | Passage anchors, then hashed-shingle occurrence counts | Anchors retrieve candidates; directional counts quantify shared textual features. |
| Do scanned pages look alike? | Per-page perceptual hashes and visual keys | Hamming distance describes visual similarity. |

For example, if a document with 100 hashed five-word occurrences shares 95 occurrences with a document containing 1,000, coverage is 95% **from the smaller document's direction** and 9.5% from the larger document's direction. Coverage counts shared hashed five-word occurrences. Retain the counts in both fingerprints for this calculation; MinHash estimates set resemblance.

## Decisions and limits

Raw-hash equality establishes byte identity. Canonical-text equality establishes equality of extracted normalized tokens. Shared templates, repeated footers, and OCR errors can affect approximate and visual scores. The application should define its review or skip policy using the relevant modality, page coverage, and extraction quality.

For a native PDF and its scanned counterpart, OCR text and page hashes provide separate retrieval routes. Their effectiveness depends on recognition and rendering quality. The available evidence and measured results are described in [Methodology](methodology.md) and [Evaluation](evaluation.md).

Fingerprinting runs locally on the CPU. DOCX/PPTX conversion uses office2pdf; scanned-text recognition uses Tesseract. The application owns storage and retrieval.
