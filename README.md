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

### With Element-Aware Node Parsing

```python
from llama_index.core.ingestion import IngestionPipeline
from llama_index.readers.kreuzberg import KreuzbergReader
from llama_index.node_parser.kreuzberg import KreuzbergNodeParser

documents = KreuzbergReader().load_data("report.pdf")

pipeline = IngestionPipeline(
    transformations=[
        KreuzbergNodeParser(),
    ]
)
nodes = pipeline.run(documents=documents)
```

## License

MIT
