# LlamaIndex Node Parser Kreuzberg

Element-aware LlamaIndex node parser for kreuzberg-extracted documents.

## Installation

```bash
pip install llama-index-node-parser-kreuzberg
```

Requires `llama-index-core>=0.13.0,<0.15`. This package does not depend on
`kreuzberg` directly — the `kreuzberg` package is a dependency of the reader
(`llama-index-readers-kreuzberg`), which is needed for producing documents with
element metadata.

## Prerequisites

> **This parser requires documents with `_kreuzberg_elements` metadata.**
> These are produced by `KreuzbergReader` configured with element-based
> extraction. Install `llama-index-readers-kreuzberg` (which brings in
> `kreuzberg`) to use the full workflow.

```python
from kreuzberg import ExtractionConfig
from llama_index.readers.kreuzberg import KreuzbergReader

reader = KreuzbergReader(
    extraction_config=ExtractionConfig(result_format="element_based")
)
documents = reader.load_data("report.pdf")
```

## Features

- Element-aware splitting — headings, paragraphs, tables, and code blocks each become a node
- Element type metadata preserved on each node (`element_type`, `page_number`, `element_index`)
- Source document relationships tracked via `NodeRelationship.SOURCE`
- Graceful degradation — documents without elements pass through with a warning
- Composes with other transformations (e.g., `SentenceSplitter` for further chunking)
- Async support via `aget_nodes_from_documents`
- Serialization support (`to_dict` / `from_dict`)

## Usage

### Basic

Full reader-to-nodes flow:

```python
from kreuzberg import ExtractionConfig
from llama_index.readers.kreuzberg import KreuzbergReader
from llama_index.node_parser.kreuzberg import KreuzbergNodeParser

reader = KreuzbergReader(
    extraction_config=ExtractionConfig(result_format="element_based")
)
documents = reader.load_data("report.pdf")

parser = KreuzbergNodeParser()
nodes = parser.get_nodes_from_documents(documents)
```

### IngestionPipeline

Chain with `SentenceSplitter` for further chunking of large elements:

```python
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

pipeline = IngestionPipeline(
    transformations=[
        KreuzbergNodeParser(),
        SentenceSplitter(chunk_size=512),  # Further split large elements
    ]
)
nodes = pipeline.run(documents=documents)
```

### VectorStoreIndex

Using the `transformations` parameter:

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(
    documents,
    transformations=[KreuzbergNodeParser()],
)
```

### Async

```python
nodes = await parser.aget_nodes_from_documents(documents)
```

## Behavior Notes

- Documents without `_kreuzberg_elements` metadata pass through unchanged with
  a warning. This is intentional — silently falling back would prevent users
  from noticing they are not getting element-aware splitting.
- Empty or whitespace-only elements are automatically skipped.
