# LlamaIndex Readers Kreuzberg

<div align="center" style="display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin: 20px 0;">
  <a href="https://pypi.org/project/llama-index-readers-kreuzberg/">
    <img src="https://img.shields.io/pypi/v/llama-index-readers-kreuzberg?label=Reader&color=007ec6" alt="Reader">
  </a>
  <a href="https://pypi.org/project/kreuzberg/">
    <img src="https://img.shields.io/pypi/v/kreuzberg?label=Kreuzberg&color=007ec6" alt="Kreuzberg">
  </a>
  <a href="https://github.com/kreuzberg-dev/llama-index-kreuzberg/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  </a>
  <a href="https://docs.kreuzberg.dev">
    <img src="https://img.shields.io/badge/docs-kreuzberg.dev-blue" alt="Documentation">
  </a>
</div>

<img width="3384" height="573" alt="Kreuzberg Banner" src="https://github.com/user-attachments/assets/1b6c6ad7-3b6d-4171-b1c9-f2026cc9deb8" />

<div align="center" style="margin-top: 20px;">
  <a href="https://discord.gg/xt9WY3GnKR">
    <img height="22" src="https://img.shields.io/badge/Discord-Join%20our%20community-7289da?logo=discord&logoColor=white" alt="Discord">
  </a>
</div>

LlamaIndex reader for 91+ file formats powered by [kreuzberg](https://github.com/kreuzberg-dev/kreuzberg)'s Rust extraction engine.

## Installation

```bash
pip install llama-index-readers-kreuzberg
```

Requires Python ≥3.10, `kreuzberg>=4.9.4`, and `llama-index-core>=0.13,<0.15`.

## Features

- **91+ file formats** -- PDF, DOCX, PPTX, XLSX, HTML, images, emails, archives, and more ([full list](https://docs.kreuzberg.dev/reference/formats/))
- **Rich metadata** -- quality scores, language detection, keywords, annotations
- **Element extraction** -- structural elements for structure-aware RAG pipelines
- **Image extraction** -- base64-encoded image data with position, format, and OCR metadata
- **Per-page splitting** -- one `Document` per page for fine-grained retrieval
- **Batch processing** -- multiple files in a single call
- **Raw bytes input** -- extract from in-memory bytes with a MIME type
- **Native async** -- true async via kreuzberg's Rust tokio runtime
- **Error tolerance** -- skip failed files with warnings, or raise on failure
- **Full serialization** -- custom `ExtractionConfig` round-trips through `to_dict()`/`from_dict()` for pipeline caching

## Usage

### Basic Extraction

```python
from llama_index.readers.kreuzberg import KreuzbergReader

reader = KreuzbergReader()
documents = reader.load_data("report.pdf")

# Each document carries rich metadata
print(documents[0].metadata["file_name"])       # "report.pdf"
print(documents[0].metadata["file_type"])        # "application/pdf"
print(documents[0].metadata["total_pages"])      # 12
print(documents[0].metadata["quality_score"])    # 0.95
print(documents[0].metadata["detected_languages"])  # ["en"]
```

### OCR Configuration

`force_ocr` is a top-level `ExtractionConfig` option. Language and backend are set on `OcrConfig`.

```python
from kreuzberg import ExtractionConfig, OcrConfig

reader = KreuzbergReader(
    extraction_config=ExtractionConfig(
        force_ocr=True,
        ocr=OcrConfig(language="deu", backend="tesseract"),
    )
)
documents = reader.load_data("scanned.pdf")
```

### Per-Page Splitting

`PageConfig` is nested inside `ExtractionConfig`. Each page becomes its own `Document`
with a `page_number` metadata field.

```python
from kreuzberg import ExtractionConfig, PageConfig

reader = KreuzbergReader(
    extraction_config=ExtractionConfig(
        pages=PageConfig(extract_pages=True),
    )
)
documents = reader.load_data("multi_page.pdf")  # One Document per page

for doc in documents:
    print(f"Page {doc.metadata['page_number']}: {doc.text[:80]}...")
```

### Element Extraction

Setting `result_format="element_based"` populates `_kreuzberg_elements` in document
metadata for structure-aware processing.

```python
from kreuzberg import ExtractionConfig

reader = KreuzbergReader(
    extraction_config=ExtractionConfig(result_format="element_based")
)
documents = reader.load_data("report.pdf")

# Structural elements available for downstream node parsers
elements = documents[0].metadata["_kreuzberg_elements"]
```

### Batch Processing

Pass a list of file paths to extract multiple files in one call.

```python
reader = KreuzbergReader()
documents = reader.load_data(["report.pdf", "slides.pptx", "data.xlsx"])
```

### Raw Bytes

Use `data=` and `mime_type=` keyword arguments to extract from in-memory bytes.

```python
reader = KreuzbergReader()

# Single bytes input
documents = reader.load_data(data=pdf_bytes, mime_type="application/pdf")

# Batch bytes input -- parallel lists of data and MIME types
documents = reader.load_data(
    data=[pdf_bytes, docx_bytes],
    mime_type=["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
)
```

### Async

`aload_data` provides native async extraction backed by kreuzberg's Rust runtime.

```python
documents = await reader.aload_data(["file1.pdf", "file2.pdf"])
```

### SimpleDirectoryReader Integration

Register `KreuzbergReader` as a file extractor for any supported extension.

```python
from llama_index.core import SimpleDirectoryReader

reader = KreuzbergReader()
sdr = SimpleDirectoryReader(
    input_dir="./documents",
    file_extractor={".pdf": reader, ".docx": reader, ".html": reader},
)
documents = sdr.load_data()

# Async variant works too
documents = await sdr.aload_data()
```

## Behavior Notes

- **Error tolerance**: By default, `raise_on_error=False` -- the reader logs warnings and skips files that fail extraction.
- **Strict mode**: Set `raise_on_error=True` to propagate extraction exceptions immediately.
- **Deterministic IDs**: Document IDs are SHA-256 hashes of the file path (or byte content) and page number, enabling stable deduplication across pipeline runs.
- **Metadata exclusion**: Large metadata fields (`_kreuzberg_elements`, `images`) are automatically excluded from LLM and embedding metadata keys to keep prompt sizes manageable.
- **Table handling**: Tables extracted by kreuzberg are appended as markdown to the document text when they are not already present in the content.
- **Serialization**: The reader fully supports `to_dict()`/`from_dict()` round-tripping, including `ExtractionConfig` with nested `OcrConfig` and `PageConfig`. This enables pipeline caching and persistence with `IngestionPipeline`.

## Metadata Reference

Each `Document` produced by `KreuzbergReader` includes these metadata fields (when available from the source document):

| Field | Type | Description |
|---|---|---|
| `file_name` | `str` | Source file name or `"bytes"` for raw bytes input |
| `file_path` | `str` | Absolute path to the source file |
| `file_type` | `str` | MIME type of the source document |
| `total_pages` | `int` | Total page count of the source document |
| `page_number` | `int` | Page number (present only with per-page splitting) |
| `quality_score` | `float` | Extraction quality score (0.0 -- 1.0) |
| `detected_languages` | `list[str]` | ISO language codes detected in the text |
| `output_format` | `str` | Format of the extracted content (`"text"`, `"markdown"`, etc.) |
| `extracted_keywords` | `list[dict]` | Keywords with text, score, and algorithm |
| `annotations` | `list[dict]` | Document annotations (comments, highlights) |
| `processing_warnings` | `list[dict]` | Warnings encountered during extraction |
| `_kreuzberg_elements` | `list` | Structural elements (with `result_format="element_based"`) |
| `images` | `list[dict]` | Base64-encoded images with position and format metadata |
