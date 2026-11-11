# Methodology

DocFGP computes a `DocumentFingerprint` for PDF, DOCX, PPTX, PNG, JPEG, TIFF, BMP, and WebP files.

## From file to evidence

| Stage | Implementation | Result |
| --- | --- | --- |
| Input validation | Format signature, file size, and Office package checks | Identified format or an explicit error |
| Office conversion | [office2pdf](https://github.com/developer0hye/office2pdf) converts DOCX/PPTX to a temporary PDF | A common page representation |
| Page extraction | [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) extracts PDF text and renders pages; Pillow decodes images | Text, page images, and extraction provenance |
| OCR when indicated | [Tesseract](https://github.com/tesseract-ocr/tesseract) runs on a bounded page image | Recognized text and mean word confidence |
| Feature construction | SHA-256, BLAKE2b, shingles, MinHash, winnowing, and perceptual hashing | A versioned, serializable fingerprint |

Pages are processed in order. Native PDF text is used when it has word tokens and valid decoded characters. OCR is attempted when native text is missing or invalid, or when an embedded image covers at least half of a page. When an image-heavy page also has native text, OCR replaces that text if its mean recognized-word confidence is at least 60 and it contributes at least three new tokens. For missing or invalid native text, accepted OCR needs confidence of at least 30. Page quality records OCR attempts, results, and warnings. These thresholds are deterministic text-selection rules.

OCR uses the models named by `ocr_language`, which defaults to English (`eng`). The package also includes the general Latin model (`script/Latin`), which OCR uses for Latin-alphabet language codes such as `deu`, `fra`, or `spa` when no installed model has that code; Fraktur, early-modern, and Vietnamese text require their own models. The optional `docfgp[languages]` extra adds the models for every language written in a non-Latin script, such as `chi_sim` or `ara`, which combine with the included models, as in `deu+ara`, and `tessdata_path` selects a custom model directory. Creating a `Fingerprinter` with a missing model raises `FingerprintError`. `parameters` records the models used, the Tesseract version, and the model revision.

## Text representation

Selected page text is normalized with Unicode NFKC, case folding, whitespace compaction, and joining of words divided by a line-break hyphen. Unicode word tokenization retains numbers and combining marks. Scripts written without spaces between words, such as Chinese, Japanese, and Thai, yield one token per letter or digit. The **canonical text** is the ordered token sequence joined with spaces. A second representation removes punctuation and spacing from canonical text before character shingling. It tolerates some OCR spacing differences while retaining letters, digits, and combining marks.

Reading order and OCR quality influence the token sequence. Canonical equality means that the *extracted and normalized tokens* match.

## Fingerprint components

| Component | Construction | Intended use |
| --- | --- | --- |
| Raw hash | SHA-256 of original bytes | Byte identity |
| Canonical hash | SHA-256 of nonempty canonical text | Exact extracted-text identity across formats |
| Word features | Consecutive five-token shingles, each hashed to 64 bits with BLAKE2b | Text resemblance and occurrence counts |
| Character features | Consecutive 13-character shingles of the tolerant text | Retrieval across some OCR differences |
| MinHash | 512 deterministic components for each shingle set | Approximate set similarity |
| LSH keys | 32 bands of 16 MinHash components per sketch | Indexed candidate retrieval |
| Passage anchors | Rightmost minimum word-shingle hash in each 32-shingle window, stored as passage lookup keys | Retrieval of shared passages |
| Page hashes | 64-bit perceptual hash per nonblank rendered page; four 16-bit lookup keys | Visual candidate retrieval |

Exact hashes use SHA-256 and text-feature identifiers use 64-bit BLAKE2b, so values are stable across Python processes. MinHash permutations are derived from fixed seed material. Empty feature sets yield empty sketches and lookup-key groups. The current result uses schema version `2.0` and algorithm version `1.5`; it also records feature parameters, per-page provenance, warnings, and a quality status.

For two documents with word-shingle sets $S_A$ and $S_B$, the fraction of equal MinHash components estimates set Jaccard similarity:

$$
J(S_A,S_B)=\frac{|S_A\cap S_B|}{|S_A\cup S_B|}.
$$

With 32 bands of 16 components, the usual independent-MinHash model gives a probability of at least one shared band of $1-(1-s^{16})^{32}$ at Jaccard similarity $s$. Empirical retrieval recall also depends on extraction quality and the implemented hash family. The sketch estimates *set* similarity and discards repetition. The underlying resemblance and containment approach is described by [Broder](https://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/broder97resemblance.pdf).

Occurrence counts retain repetition. If $c_A(h)$ is the number of occurrences of shingle hash $h$ in A, the application can calculate

$$
I(A,B)=\sum_h\min(c_A(h),c_B(h)),\qquad
C(A\mid B)=\frac{I(A,B)}{\sum_h c_A(h)}.
$$

Coverage is directional because $C(A\mid B)$ and $C(B\mid A)$ use different denominators. A value of 0.95 means that 95% of A's **hashed five-word shingle occurrences** also occur in B, up to hash collision. Set `include_shingle_counts=True` to retain the counts for this calculation.

Winnowing provides a separate route to partial matches. A contiguous passage of at least 36 identical normalized words contains a complete window of 32 five-word shingles and therefore shares an anchor under this selection rule. Shorter or edited passages depend on the anchors they retain. Repeated boilerplate can also share anchors. This method measures shared surface text. See the [winnowing paper](https://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/p76-schleimer.pdf).

Page perceptual hashes use a 32×32 grayscale discrete cosine transform. Their Hamming distance describes visual similarity independently of text coverage. Candidate retrieval through visual keys requires at least one matching 16-bit chunk; the application can then compare complete page hashes.

## Limits and reproducibility

Default limits are 50 MB per file, 200 pages, 8 million decoded pixels per page, 2 million extracted characters, and 200,000 distinct word or character shingles. PDF rendering uses 96 DPI for page hashes and up to 180 DPI for OCR, subject to the pixel limit; standalone images retain their source dimensions within that limit. Office conversion and OCR have a 60-second timeout; the document has a 180-second cooperative deadline. Configuration can change these limits.

`quality.status` identifies the available text and visual evidence. `insufficient_evidence` retains the raw hash and has empty approximate lookup-key groups. Invalid documents and exceeded processing limits raise `FingerprintError`; OCR failure can leave a warning and a visual-only result. Temporary conversion files are removed after the call. The cooperative deadline is checked between in-process decoding stages; services accepting untrusted uploads should also apply operating-system resource limits.

Fixed feature parameters and deterministic serialization support repeatable results. Changes in office2pdf, PDFium, Pillow, Tesseract, fonts, or OCR language can change extracted evidence, so experiments should record those versions and settings.
