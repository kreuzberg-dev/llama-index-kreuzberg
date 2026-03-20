"""Integration tests for KreuzbergNodeParser pipeline compatibility."""

from unittest.mock import patch

import pytest
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

from llama_index.node_parser.kreuzberg import KreuzbergNodeParser
from tests.conftest import make_kreuzberg_document


def test_ingestion_pipeline_round_trip() -> None:
    doc = make_kreuzberg_document()
    pipeline = IngestionPipeline(transformations=[KreuzbergNodeParser()])

    nodes = pipeline.run(documents=[doc])

    assert len(nodes) == 3
    assert nodes[0].metadata["element_type"] == "title"


def test_pipeline_composition_with_sentence_splitter() -> None:
    doc = make_kreuzberg_document()
    pipeline = IngestionPipeline(
        transformations=[
            KreuzbergNodeParser(),
            SentenceSplitter(chunk_size=1024, chunk_overlap=0),
        ],
    )

    nodes = pipeline.run(documents=[doc])

    assert len(nodes) >= 3
    for node in nodes:
        assert node.text


def test_vector_store_index_integration() -> None:
    doc = make_kreuzberg_document()

    with patch("llama_index.core.VectorStoreIndex.build_index_from_nodes"):
        Settings.embed_model = MockEmbedding(embed_dim=8)
        Settings.llm = None
        index = VectorStoreIndex.from_documents(
            [doc],
            transformations=[KreuzbergNodeParser()],
        )
        assert index is not None


def test_serialization_round_trip() -> None:
    parser = KreuzbergNodeParser(include_prev_next_rel=False)

    data = parser.to_dict()
    # id_func serializes to a dict that can't round-trip to Callable —
    # this is a known LlamaIndex-wide limitation, not specific to us
    data.pop("id_func", None)
    restored = KreuzbergNodeParser.from_dict(data)

    assert restored.include_prev_next_rel is False
    assert restored.class_name() == "KreuzbergNodeParser"


@pytest.mark.asyncio
async def test_acall_produces_same_results_as_sync() -> None:
    doc = make_kreuzberg_document()
    parser = KreuzbergNodeParser()

    sync_nodes = parser([doc])
    async_nodes = await parser.acall([doc])

    assert len(async_nodes) == len(sync_nodes)
    for sync_node, async_node in zip(sync_nodes, async_nodes):
        assert sync_node.text == async_node.text
        assert sync_node.metadata["element_type"] == async_node.metadata["element_type"]
        assert "_kreuzberg_elements" not in async_node.metadata
