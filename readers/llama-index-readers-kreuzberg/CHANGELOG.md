# Changelog

All notable changes to llama-index-readers-kreuzberg will be documented in this file.

## [Unreleased]

## [0.1.0] — 2026-03-20

Initial release.

### Added

- `KreuzbergReader` — LlamaIndex reader for 88+ document formats powered by kreuzberg's Rust extraction engine
- Sync and true async extraction via `load_data` / `aload_data` and lazy variants
- File path and raw bytes input, including batch extraction (`list[Path]`, `list[bytes]`)
- Per-page document splitting when kreuzberg returns page-level results
- Maximalist metadata: `file_type`, `total_pages`, `quality_score`, `detected_languages`, `output_format`, `processing_warnings`, `extracted_keywords`, `annotations`
- Element-based extraction support (`_kreuzberg_elements` metadata) for downstream `KreuzbergNodeParser`
- Image extraction with base64 serialization and per-page filtering
- Table appending: markdown tables merged into page content when not already present
- SHA-256 deterministic document IDs for deduplication
- Full `ExtractionConfig` serialization round-trip via `dict_to_config` / `config_to_json` for pipeline persistence
- Forward-compatible config reconstruction: fields from newer kreuzberg versions are silently ignored
- `raise_on_error` flag for controlling extraction failure behavior (log-and-skip vs propagate)
- kreuzberg v4.5.0 optional config support (`EmailConfig`, `AccelerationConfig`, `LayoutDetectionConfig`)
