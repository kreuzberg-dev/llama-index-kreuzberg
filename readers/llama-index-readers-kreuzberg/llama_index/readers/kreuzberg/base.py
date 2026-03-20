"""KreuzbergReader — LlamaIndex reader for 88+ document formats.

Wraps kreuzberg's Rust-core extraction engine with true async support,
maximalist metadata, and lossless pipeline persistence.
"""

import base64
import hashlib
import json
import logging
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from llama_index.readers.kreuzberg._config import dict_to_config
from pydantic import Field, field_serializer, field_validator

from kreuzberg import (
    ExtractionConfig,
    batch_extract_bytes,
    batch_extract_bytes_sync,
    batch_extract_files,
    batch_extract_files_sync,
    config_to_json,
    extract_bytes,
    extract_bytes_sync,
    extract_file,
    extract_file_sync,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _ExtractionTask:
    """Validated extraction plan — what to extract, not how."""

    kind: Literal["file", "file_batch", "bytes", "bytes_batch"]
    paths: tuple[Path, ...] = ()
    data_list: tuple[bytes, ...] = ()
    mime_types: tuple[str, ...] = ()


def _build_metadata(  # noqa: C901
    result: Any,
    file_path: Path | None = None,
    source: str | None = None,
    extra_info: dict[str, Any] | None = None,
    page_number: int | None = None,
) -> dict[str, Any]:
    """Flatten ExtractionResult into a metadata dict."""
    meta: dict[str, Any] = {}

    if file_path is not None:
        meta["file_name"] = Path(file_path).name
        meta["file_path"] = str(file_path)
    elif source is not None:
        meta["file_name"] = source
        meta["file_path"] = source

    meta["file_type"] = result.mime_type
    meta["total_pages"] = result.get_page_count()

    if page_number is not None:
        meta["page_number"] = page_number

    kreuzberg_meta = result.metadata
    if isinstance(kreuzberg_meta, dict):
        meta.update({k: v for k, v in kreuzberg_meta.items() if v is not None})

    if result.quality_score is not None:
        meta["quality_score"] = result.quality_score
    if result.detected_languages is not None:
        meta["detected_languages"] = result.detected_languages
    if result.output_format is not None:
        meta["output_format"] = result.output_format
    if result.processing_warnings:
        meta["processing_warnings"] = [
            {"source": w.source, "message": w.message} if hasattr(w, "source") else str(w)
            for w in result.processing_warnings
        ]
    if result.extracted_keywords:
        meta["extracted_keywords"] = [
            {"text": kw.text, "score": kw.score, "algorithm": kw.algorithm} if hasattr(kw, "text") else kw
            for kw in result.extracted_keywords
        ]
    if result.annotations:
        meta["annotations"] = [
            {
                "annotation_type": a.annotation_type,
                "content": a.content,
                "page_number": a.page_number,
            }
            if hasattr(a, "annotation_type")
            else a
            for a in result.annotations
        ]

    if extra_info:
        meta.update(extra_info)

    return meta


def _generate_doc_id(
    *,
    file_path: Path | None = None,
    data: bytes | None = None,
    page_number: int | None = None,
) -> str:
    """Generate a deterministic document ID via SHA-256."""
    if file_path is None and data is None:
        msg = "Either file_path or data must be provided"
        raise ValueError(msg)
    hasher = hashlib.sha256()
    if file_path is not None:
        hasher.update(str(file_path.resolve()).encode())
    elif data is not None:
        hasher.update(data)
    if page_number is not None:
        hasher.update(str(page_number).encode())
    return hasher.hexdigest()


class KreuzbergReader(BasePydanticReader):
    """Reader for 88+ document formats powered by kreuzberg's Rust extraction engine.

    Supports file paths, raw bytes, batch input, per-page splitting,
    and true async via Rust tokio.

    Note:
        This is a local-only reader (``is_remote = False``). Remote/virtual
        filesystems (the ``fs`` parameter used by ``SimpleDirectoryReader``)
        are not supported.

    """

    is_remote: bool = False
    raise_on_error: bool = Field(
        default=False,
        description="If True, propagate kreuzberg exceptions. If False, log warnings and skip failed files.",
    )
    extraction_config: ExtractionConfig | None = Field(
        default=None,
        description="Full kreuzberg ExtractionConfig for controlling output format, "
        "OCR, image extraction, and all other extraction options.",
    )

    @classmethod
    def class_name(cls) -> str:
        """Return the canonical class name used for serialization."""
        return "KreuzbergReader"

    @field_validator("extraction_config", mode="before")
    @classmethod
    def _validate_config(cls, v: Any) -> ExtractionConfig | None:
        if v is None:
            return None
        if isinstance(v, dict):
            return dict_to_config(v)
        if isinstance(v, ExtractionConfig):
            return v
        msg = f"Expected ExtractionConfig, dict, or None, got {type(v)}"
        raise ValueError(msg)

    @field_serializer("extraction_config")
    def _serialize_config(self, v: ExtractionConfig | None) -> dict[str, Any] | None:
        if v is None:
            return None
        return dict[str, Any](json.loads(config_to_json(v)))

    def _build_config(self) -> ExtractionConfig:
        """Return the ExtractionConfig to use for extraction."""
        return self.extraction_config or ExtractionConfig()

    def load_data(  # noqa: D102
        self,
        file_path: str | Path | list[str] | list[Path] | None = None,
        extra_info: dict[str, Any] | None = None,
        *,
        data: bytes | list[bytes] | None = None,
        mime_type: str | list[str] | None = None,
    ) -> list[Document]:
        return list(
            self.lazy_load_data(
                file_path=file_path,
                extra_info=extra_info,
                data=data,
                mime_type=mime_type,
            )
        )

    def lazy_load_data(  # noqa: D102
        self,
        file_path: str | Path | list[str] | list[Path] | None = None,
        extra_info: dict[str, Any] | None = None,
        *,
        data: bytes | list[bytes] | None = None,
        mime_type: str | list[str] | None = None,
    ) -> Iterable[Document]:
        config = self._build_config()
        results_with_source = self._extract_sync(file_path=file_path, data=data, mime_type=mime_type, config=config)
        yield from self._results_to_documents(results_with_source, extra_info)

    def _extract_sync(  # noqa: C901, PLR0912
        self,
        *,
        file_path: str | Path | list[str] | list[Path] | None = None,
        data: bytes | list[bytes] | None = None,
        mime_type: str | list[str] | None = None,
        config: ExtractionConfig,
    ) -> list[tuple[Any, Path | None, bytes | None]]:
        results: list[tuple[Any, Path | None, bytes | None]] = []

        if file_path is not None:
            paths = [Path(p) for p in file_path] if isinstance(file_path, list) else [Path(file_path)]
            if len(paths) == 1:
                result = self._safe_extract(
                    lambda: extract_file_sync(paths[0], config=config),
                    source=str(paths[0]),
                )
                if result is not None:
                    results.append((result, paths[0], None))
            else:
                batch_results = self._safe_batch_extract(
                    lambda: batch_extract_files_sync(paths, config=config),
                    sources=[str(p) for p in paths],
                )
                for r, p in zip(batch_results, paths, strict=True):
                    if r is not None:
                        results.append((r, p, None))
        elif data is not None:
            if isinstance(data, list):
                if not isinstance(mime_type, list) or len(data) != len(mime_type):
                    msg = "data and mime_type must be parallel lists of equal length"
                    raise ValueError(msg)
                batch_results = self._safe_batch_extract(
                    lambda: batch_extract_bytes_sync(data, mime_type, config=config),
                    sources=[f"bytes[{i}]" for i in range(len(data))],
                )
                for r, d in zip(batch_results, data, strict=True):
                    if r is not None:
                        results.append((r, None, d))
            else:
                if mime_type is None or isinstance(mime_type, list):
                    msg = "mime_type must be a string for single bytes input"
                    raise ValueError(msg)
                result = self._safe_extract(
                    lambda: extract_bytes_sync(data, mime_type, config=config),
                    source="bytes",
                )
                if result is not None:
                    results.append((result, None, data))
        else:
            msg = "Either file_path or data must be provided"
            raise ValueError(msg)

        return results

    def _safe_extract(self, fn: Any, source: str) -> Any | None:
        try:
            return fn()
        except Exception:
            if self.raise_on_error:
                raise
            logger.warning("Failed to extract %s", source, exc_info=True)
            return None

    def _safe_batch_extract(self, fn: Any, sources: list[str]) -> list[Any | None]:
        try:
            return list(fn())
        except Exception:
            if self.raise_on_error:
                raise
            logger.warning("Batch extraction failed for %s", ", ".join(sources), exc_info=True)
            return [None] * len(sources)

    def _results_to_documents(
        self,
        results_with_source: list[tuple[Any, Path | None, bytes | None]],
        extra_info: dict[str, Any] | None = None,
    ) -> Iterable[Document]:
        """Yield Documents from extraction results, one per page when pages are present."""
        for result, file_path, data in results_with_source:
            if result.pages:
                for page in result.pages:
                    page_num = page.page_number if hasattr(page, "page_number") else 1
                    page_content = page.content if hasattr(page, "content") else ""
                    content = self._append_tables_if_needed(
                        page_content,
                        page.tables if hasattr(page, "tables") else [],
                    )
                    meta = _build_metadata(
                        result=result,
                        file_path=file_path,
                        source="bytes" if data is not None else None,
                        extra_info=extra_info,
                        page_number=page_num,
                    )
                    excluded_keys: list[str] = []
                    if result.elements is not None:
                        meta["_kreuzberg_elements"] = result.elements
                        excluded_keys = ["_kreuzberg_elements"]
                    if result.images:
                        meta["images"] = self._serialize_images(result.images, page_number=page_num)
                        excluded_keys.append("images")
                    yield Document(
                        text=content,
                        id_=_generate_doc_id(file_path=file_path, data=data, page_number=page_num),
                        metadata=meta,
                        excluded_llm_metadata_keys=excluded_keys,
                        excluded_embed_metadata_keys=excluded_keys,
                    )
            else:
                content = self._append_tables_if_needed(result.content, result.tables)
                meta = _build_metadata(
                    result=result,
                    file_path=file_path,
                    source="bytes" if data is not None else None,
                    extra_info=extra_info,
                )
                excluded_keys = []
                if result.elements is not None:
                    meta["_kreuzberg_elements"] = result.elements
                    excluded_keys = ["_kreuzberg_elements"]
                if result.images:
                    meta["images"] = self._serialize_images(result.images)
                    excluded_keys.append("images")
                yield Document(
                    text=content,
                    id_=_generate_doc_id(file_path=file_path, data=data),
                    metadata=meta,
                    excluded_llm_metadata_keys=excluded_keys,
                    excluded_embed_metadata_keys=excluded_keys,
                )

    @staticmethod
    def _append_tables_if_needed(content: str, tables: list[Any]) -> str:
        """Append table markdown to content when tables are not already included."""
        if not tables:
            return content
        for table in tables:
            table_md = table.markdown if hasattr(table, "markdown") else str(table)
            if table_md and table_md.strip() not in content:
                content = content.rstrip() + "\n\n" + table_md
        return content

    @staticmethod
    def _serialize_images(images: list[Any], page_number: int | None = None) -> list[dict[str, Any]]:
        """Serialize image objects to JSON-safe dicts, filtering by page when given."""
        serialized = []
        for img in images:
            if page_number is not None:
                img_page = getattr(img, "page_number", None)
                if img_page is not None and img_page != page_number:
                    continue
            entry: dict[str, Any] = {
                "format": getattr(img, "format", "unknown"),
                "image_index": getattr(img, "image_index", 0),
                "page_number": getattr(img, "page_number", 0),
                "width": getattr(img, "width", 0),
                "height": getattr(img, "height", 0),
                "colorspace": getattr(img, "colorspace", ""),
                "bits_per_component": getattr(img, "bits_per_component", 0),
                "is_mask": getattr(img, "is_mask", False),
                "description": getattr(img, "description", ""),
            }
            img_data = getattr(img, "data", None)
            if img_data is not None:
                entry["data"] = base64.b64encode(img_data).decode("ascii")
            bbox = getattr(img, "bounding_box", None)
            if bbox is not None:
                entry["bounding_box"] = bbox
            ocr_result = getattr(img, "ocr_result", None)
            if ocr_result is not None:
                entry["ocr_result"] = ocr_result.content if hasattr(ocr_result, "content") else str(ocr_result)
            serialized.append(entry)
        return serialized

    async def aload_data(  # noqa: D102
        self,
        file_path: str | Path | list[str] | list[Path] | None = None,
        extra_info: dict[str, Any] | None = None,
        *,
        data: bytes | list[bytes] | None = None,
        mime_type: str | list[str] | None = None,
    ) -> list[Document]:
        return [
            doc
            async for doc in self.alazy_load_data(
                file_path=file_path, extra_info=extra_info, data=data, mime_type=mime_type
            )
        ]

    async def alazy_load_data(  # type: ignore[override]  # noqa: D102
        self,
        file_path: str | Path | list[str] | list[Path] | None = None,
        extra_info: dict[str, Any] | None = None,
        *,
        data: bytes | list[bytes] | None = None,
        mime_type: str | list[str] | None = None,
    ) -> AsyncIterator[Document]:
        config = self._build_config()
        results_with_source = await self._extract_async(
            file_path=file_path, data=data, mime_type=mime_type, config=config
        )
        for doc in self._results_to_documents(results_with_source, extra_info):
            yield doc

    async def _extract_async(  # noqa: C901, PLR0912
        self,
        *,
        file_path: str | Path | list[str] | list[Path] | None = None,
        data: bytes | list[bytes] | None = None,
        mime_type: str | list[str] | None = None,
        config: ExtractionConfig,
    ) -> list[tuple[Any, Path | None, bytes | None]]:
        results: list[tuple[Any, Path | None, bytes | None]] = []

        if file_path is not None:
            paths = [Path(p) for p in file_path] if isinstance(file_path, list) else [Path(file_path)]
            if len(paths) == 1:
                result = await self._safe_extract_async(extract_file(paths[0], config=config), source=str(paths[0]))
                if result is not None:
                    results.append((result, paths[0], None))
            else:
                batch_results = await self._safe_batch_extract_async(
                    batch_extract_files(paths, config=config),
                    sources=[str(p) for p in paths],
                )
                for r, p in zip(batch_results, paths, strict=True):
                    if r is not None:
                        results.append((r, p, None))
        elif data is not None:
            if isinstance(data, list):
                if not isinstance(mime_type, list) or len(data) != len(mime_type):
                    msg = "data and mime_type must be parallel lists of equal length"
                    raise ValueError(msg)
                batch_results = await self._safe_batch_extract_async(
                    batch_extract_bytes(data, mime_type, config=config),
                    sources=[f"bytes[{i}]" for i in range(len(data))],
                )
                for r, d in zip(batch_results, data, strict=True):
                    if r is not None:
                        results.append((r, None, d))
            else:
                if mime_type is None or isinstance(mime_type, list):
                    msg = "mime_type must be a string for single bytes input"
                    raise ValueError(msg)
                result = await self._safe_extract_async(extract_bytes(data, mime_type, config=config), source="bytes")
                if result is not None:
                    results.append((result, None, data))
        else:
            msg = "Either file_path or data must be provided"
            raise ValueError(msg)

        return results

    async def _safe_extract_async(self, coro: Any, source: str) -> Any | None:
        try:
            return await coro
        except Exception:
            if self.raise_on_error:
                raise
            logger.warning("Failed to extract %s", source, exc_info=True)
            return None

    async def _safe_batch_extract_async(self, coro: Any, sources: list[str]) -> list[Any | None]:
        try:
            return list(await coro)
        except Exception:
            if self.raise_on_error:
                raise
            logger.warning("Batch extraction failed for %s", ", ".join(sources), exc_info=True)
            return [None] * len(sources)
