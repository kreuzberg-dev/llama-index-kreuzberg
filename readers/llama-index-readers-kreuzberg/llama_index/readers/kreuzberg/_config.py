"""ExtractionConfig dict <-> object reconstruction for pipeline persistence.

Kreuzberg's ExtractionConfig is a PyO3 class with no from_dict() method.
These helpers reconstruct typed config objects from serialized dicts,
enabling lossless to_dict() / from_dict() round-trips.

PyO3 classes do not expose constructor type hints via get_type_hints(), so
sub-sub-config field mappings are declared as static dicts rather than
derived from annotations.
"""

import functools
from typing import Any

from kreuzberg import (
    ChunkingConfig,
    EmbeddingConfig,
    ExtractionConfig,
    HierarchyConfig,
    ImageExtractionConfig,
    ImagePreprocessingConfig,
    KeywordConfig,
    LanguageDetectionConfig,
    OcrConfig,
    PageConfig,
    PdfConfig,
    PostProcessorConfig,
    RakeParams,
    TesseractConfig,
    TokenReductionConfig,
    YakeParams,
)

# Top-level sub-config fields on ExtractionConfig: field_name -> config class.
# Only fields that hold config objects (not scalars or lists) are listed here.
_TOP_LEVEL_CONFIGS: dict[str, type] = {
    "chunking": ChunkingConfig,
    "images": ImageExtractionConfig,
    "keywords": KeywordConfig,
    "language_detection": LanguageDetectionConfig,
    "ocr": OcrConfig,
    "pages": PageConfig,
    "pdf_options": PdfConfig,
    "postprocessor": PostProcessorConfig,
    "token_reduction": TokenReductionConfig,
}

# v4.5.0 additions — guarded so the reader works with both 4.4.x and 4.5.x.
try:
    from kreuzberg import EmailConfig

    _TOP_LEVEL_CONFIGS["email"] = EmailConfig
except ImportError:
    pass

try:
    from kreuzberg import AccelerationConfig

    _TOP_LEVEL_CONFIGS["acceleration"] = AccelerationConfig
except ImportError:
    pass

try:
    from kreuzberg import LayoutDetectionConfig

    _TOP_LEVEL_CONFIGS["layout"] = LayoutDetectionConfig
except ImportError:
    pass

# Nested config fields on sub-config classes: (class, field_name) -> inner class.
# Required because PyO3 classes reject raw dicts for typed constructor arguments.
_NESTED_FIELD_MAP: dict[tuple[type, str], type] = {
    (OcrConfig, "tesseract_config"): TesseractConfig,
    (TesseractConfig, "preprocessing"): ImagePreprocessingConfig,
    (PdfConfig, "hierarchy"): HierarchyConfig,
    (ChunkingConfig, "embedding"): EmbeddingConfig,
    (KeywordConfig, "rake_params"): RakeParams,
    (KeywordConfig, "yake_params"): YakeParams,
}


@functools.lru_cache(maxsize=32)
def _known_fields(cls: type) -> frozenset[str]:
    """Return the set of field names accepted by a PyO3 config class constructor.

    Uses a default instance's dir() output as a proxy for constructor parameters,
    since get_type_hints() returns an empty dict for PyO3 classes.
    """
    try:
        return frozenset(a for a in dir(cls()) if not a.startswith("_"))
    except Exception:  # noqa: BLE001 — PyO3 classes may raise arbitrary exceptions
        return frozenset()


def _reconstruct(cls: type, d: dict[str, Any]) -> Any:
    """Reconstruct a PyO3 config class from a dict.

    Filters out fields not accepted by the constructor (config_to_json can
    serialize fields that aren't valid constructor arguments), then handles
    sub-sub-configs by looking up each (cls, field_name) pair in the static
    _NESTED_FIELD_MAP and recursing when the value is a non-None dict.
    """
    accepted = _known_fields(cls)
    kwargs: dict[str, Any] = {}
    for key, value in d.items():
        if accepted and key not in accepted:
            continue
        if isinstance(value, dict):
            inner_cls = _NESTED_FIELD_MAP.get((cls, key))
            if inner_cls is not None:
                kwargs[key] = _reconstruct(inner_cls, value)
                continue
        kwargs[key] = value
    return cls(**kwargs)


def dict_to_config(d: dict[str, Any]) -> ExtractionConfig:
    """Reconstruct an ExtractionConfig from a serialized dict.

    Converts nested dicts to their typed PyO3 counterparts, then constructs
    ExtractionConfig. Fields present in the dict that are not recognised by
    the installed kreuzberg version are silently ignored so that configs
    serialized with a newer kreuzberg can still be loaded.

    Args:
        d: A dict produced by ``json.loads(config_to_json(config))``, or any
           partial dict containing ExtractionConfig field values.

    Returns:
        A fully typed ExtractionConfig instance.

    """
    accepted = _known_fields(ExtractionConfig)
    kwargs: dict[str, Any] = {}
    for key, value in d.items():
        # Skip fields unknown to the installed kreuzberg version.
        if accepted and key not in accepted:
            continue
        if isinstance(value, dict) and key in _TOP_LEVEL_CONFIGS:
            kwargs[key] = _reconstruct(_TOP_LEVEL_CONFIGS[key], value)
        else:
            kwargs[key] = value
    return ExtractionConfig(**kwargs)
