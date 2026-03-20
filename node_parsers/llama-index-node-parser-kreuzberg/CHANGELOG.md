# Changelog

All notable changes to llama-index-node-parser-kreuzberg will be documented in this file.

## [Unreleased]

## [0.1.0] — 2026-03-20

Initial release.

### Added

- `KreuzbergNodeParser` — element-aware node parser for kreuzberg-extracted documents
- Converts `_kreuzberg_elements` metadata into individual `TextNode` objects preserving document structure (titles, headings, paragraphs, tables, etc.)
- Per-element metadata: `element_type`, `page_number`, `element_index`
- Source relationship linking each `TextNode` back to its parent `Document`
- Automatic `_kreuzberg_elements` stripping from child nodes to avoid embedding the raw element list
- Empty/whitespace element filtering
- Graceful passthrough for documents without `_kreuzberg_elements` (warns and returns unchanged)
- Custom `id_func` support for deterministic node IDs
- Sync and async `get_nodes_from_documents` / `aget_nodes_from_documents`
- Compatible with `IngestionPipeline`, `VectorStoreIndex`, and chaining with `SentenceSplitter`
- Full serialization round-trip via `to_dict` / `from_dict`
