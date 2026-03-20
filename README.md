# llama-index-kreuzberg

LlamaIndex integrations for [kreuzberg](https://github.com/kreuzberg-dev/kreuzberg) — a Rust-core document intelligence library supporting 88+ file formats with OCR, layout detection, and element-based extraction.

## Packages

| Package | Description | PyPI |
|---------|-------------|------|
| `llama-index-readers-kreuzberg` | Reader that converts documents into LlamaIndex Documents with rich metadata and optional element extraction | [![PyPI](https://img.shields.io/pypi/v/llama-index-readers-kreuzberg)](https://pypi.org/p/llama-index-readers-kreuzberg) |
| `llama-index-node-parser-kreuzberg` | Element-aware node parser that maps structural elements (headings, paragraphs, tables, code blocks) to TextNodes with type metadata and sequential relationships | [![PyPI](https://img.shields.io/pypi/v/llama-index-node-parser-kreuzberg)](https://pypi.org/p/llama-index-node-parser-kreuzberg) |

## Installation

```bash
pip install llama-index-readers-kreuzberg

# Optional: element-aware node parsing
pip install llama-index-node-parser-kreuzberg
```

## Quick Start

```python
from llama_index.readers.kreuzberg import KreuzbergReader

reader = KreuzbergReader()
documents = reader.load_data("report.pdf")
```

## How They Work Together

The **reader** extracts files into LlamaIndex `Document` objects. The **node parser** splits those documents into semantic `TextNode` objects based on structural elements (headings, paragraphs, tables, code blocks). They are independent packages but designed to complement each other.

The bridge between them is `ExtractionConfig(result_format="element_based")`. When the reader is configured this way, it produces `_kreuzberg_elements` metadata that the node parser consumes for structure-aware splitting. Without this config, the node parser will pass documents through unchanged with a warning.

```python
from kreuzberg import ExtractionConfig
from llama_index.core.ingestion import IngestionPipeline
from llama_index.readers.kreuzberg import KreuzbergReader
from llama_index.node_parser.kreuzberg import KreuzbergNodeParser

# Extract with element-based format for structure-aware processing
reader = KreuzbergReader(
    extraction_config=ExtractionConfig(result_format="element_based")
)
documents = reader.load_data("report.pdf")

# Element-aware pipeline
pipeline = IngestionPipeline(
    transformations=[
        KreuzbergNodeParser(),
    ]
)
nodes = pipeline.run(documents=documents)
```

**When to use what:**

- **Reader alone** with built-in splitters (e.g. `SentenceSplitter`): simpler setup, text-level chunking.
- **Reader + node parser**: structure-aware chunking with element types preserved for filtering and retrieval.

## License

MIT
