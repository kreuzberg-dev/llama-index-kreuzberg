"""Tests for ExtractionConfig dict <-> object reconstruction."""

from __future__ import annotations

import json

from kreuzberg import (
    ExtractionConfig,
    HierarchyConfig,
    ImageExtractionConfig,
    OcrConfig,
    PageConfig,
    PdfConfig,
    TesseractConfig,
    config_to_json,
)

from llama_index.readers.kreuzberg._config import dict_to_config


class TestDictToConfig:
    """Tests for dict_to_config reconstruction."""

    def test_empty_dict_returns_default_config(self) -> None:
        config = dict_to_config({})
        assert isinstance(config, ExtractionConfig)

    def test_flat_fields_reconstructed(self) -> None:
        config = dict_to_config({"force_ocr": True, "output_format": "markdown"})
        assert config.force_ocr is True
        assert config.output_format == "markdown"

    def test_nested_ocr_config_reconstructed(self) -> None:
        config = dict_to_config({
            "ocr": {"backend": "paddleocr", "language": "deu"},
        })
        assert isinstance(config.ocr, OcrConfig)
        assert config.ocr.backend == "paddleocr"
        assert config.ocr.language == "deu"


class TestNestedReconstruction:
    """Tests for recursive sub-sub-config reconstruction."""

    def test_pdf_with_hierarchy_config(self) -> None:
        config = dict_to_config({
            "pdf_options": {
                "extract_images": True,
                "hierarchy": {"enabled": True, "k_clusters": 4},
            },
        })
        assert isinstance(config.pdf_options, PdfConfig)
        assert isinstance(config.pdf_options.hierarchy, HierarchyConfig)
        assert config.pdf_options.hierarchy.k_clusters == 4

    def test_ocr_with_tesseract_config(self) -> None:
        config = dict_to_config({
            "ocr": {
                "backend": "tesseract",
                "tesseract_config": {
                    "psm": 6,
                    "oem": 1,
                },
            },
        })
        assert isinstance(config.ocr.tesseract_config, TesseractConfig)
        assert config.ocr.tesseract_config.psm == 6

    def test_page_config_reconstructed(self) -> None:
        config = dict_to_config({
            "pages": {"extract_pages": True, "insert_page_markers": True},
        })
        assert isinstance(config.pages, PageConfig)
        assert config.pages.extract_pages is True


class TestRoundTrip:
    """Tests for serialize -> deserialize round-trip via config_to_json."""

    def test_config_round_trip(self) -> None:
        original = ExtractionConfig(
            output_format="markdown",
            force_ocr=True,
            ocr=OcrConfig(backend="tesseract", language="deu"),
            pages=PageConfig(extract_pages=True),
        )
        serialized = json.loads(config_to_json(original))
        reconstructed = dict_to_config(serialized)

        assert reconstructed.output_format == "markdown"
        assert reconstructed.force_ocr is True
        assert reconstructed.ocr.backend == "tesseract"
        assert reconstructed.ocr.language == "deu"
        assert reconstructed.pages.extract_pages is True

    def test_default_config_round_trip(self) -> None:
        original = ExtractionConfig()
        serialized = json.loads(config_to_json(original))
        reconstructed = dict_to_config(serialized)
        assert isinstance(reconstructed, ExtractionConfig)
