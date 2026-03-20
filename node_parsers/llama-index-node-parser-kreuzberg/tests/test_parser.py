"""Tests for KreuzbergNodeParser."""

from __future__ import annotations

from llama_index.core.schema import Document, TextNode

from llama_index.node_parser.kreuzberg import KreuzbergNodeParser
from tests.conftest import make_element, make_kreuzberg_document


class TestParseNodes:
    """Unit tests for _parse_nodes() core behavior."""

    def test_elements_produce_correct_text_nodes(self) -> None:
        """Test #1: Core element parsing — elements produce TextNodes with correct text and metadata."""
        doc = make_kreuzberg_document()
        parser = KreuzbergNodeParser()

        nodes = parser.get_nodes_from_documents([doc])

        assert len(nodes) == 3
        # First node: title
        assert nodes[0].text == "Document Title"
        assert nodes[0].metadata["element_type"] == "title"
        assert nodes[0].metadata["page_number"] == 1
        assert nodes[0].metadata["element_index"] == 0
        # Second node: narrative_text
        assert nodes[1].text == "First paragraph."
        assert nodes[1].metadata["element_type"] == "narrative_text"
        # Third node: table
        assert nodes[2].text == "| A | B |\n| 1 | 2 |"
        assert nodes[2].metadata["element_type"] == "table"
        assert nodes[2].metadata["page_number"] == 2
