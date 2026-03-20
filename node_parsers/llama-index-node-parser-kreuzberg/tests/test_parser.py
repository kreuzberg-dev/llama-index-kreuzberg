"""Unit tests for KreuzbergNodeParser."""

from llama_index.core.schema import BaseNode, Document
from llama_index.node_parser.kreuzberg import KreuzbergNodeParser

from tests.conftest import make_element, make_kreuzberg_document


def test_elements_produce_correct_text_nodes() -> None:
    doc = make_kreuzberg_document()
    parser = KreuzbergNodeParser()

    nodes = parser.get_nodes_from_documents([doc])

    assert len(nodes) == 3
    assert nodes[0].text == "Document Title"
    assert nodes[0].metadata["element_type"] == "title"
    assert nodes[0].metadata["page_number"] == 1
    assert nodes[0].metadata["element_index"] == 0
    assert nodes[1].text == "First paragraph."
    assert nodes[1].metadata["element_type"] == "narrative_text"
    assert nodes[2].text == "| A | B |\n| 1 | 2 |"
    assert nodes[2].metadata["element_type"] == "table"
    assert nodes[2].metadata["page_number"] == 2


def test_source_relationship_points_to_parent() -> None:
    doc = make_kreuzberg_document()
    parser = KreuzbergNodeParser()

    nodes = parser.get_nodes_from_documents([doc])

    for node in nodes:
        assert node.source_node is not None
        assert node.source_node.node_id == doc.node_id


def test_empty_and_whitespace_elements_skipped() -> None:
    elements = [
        make_element(text="Real content", element_index=0),
        make_element(text="", element_index=1),
        make_element(text="   ", element_index=2),
        make_element(text="\n\t", element_index=3),
        make_element(text="Also real", element_index=4),
    ]
    doc = make_kreuzberg_document(elements=elements)
    parser = KreuzbergNodeParser()

    nodes = parser.get_nodes_from_documents([doc])

    assert len(nodes) == 2
    assert nodes[0].text == "Real content"
    assert nodes[1].text == "Also real"


def test_missing_elements_warns_and_passes_through() -> None:
    doc = Document(text="Plain text", id_="plain-001", metadata={"file_path": "/tmp/test.txt"})
    parser = KreuzbergNodeParser()

    nodes = parser.get_nodes_from_documents([doc])

    assert len(nodes) == 1
    assert nodes[0].text == "Plain text"
    assert nodes[0].id_ == "plain-001"


def test_empty_elements_list_warns_and_passes_through() -> None:
    doc = make_kreuzberg_document(elements=[])
    parser = KreuzbergNodeParser()

    nodes = parser.get_nodes_from_documents([doc])

    assert len(nodes) == 1
    assert nodes[0].text == "Full document text."


def test_kreuzberg_elements_stripped_from_children_not_passthrough() -> None:
    kreuzberg_doc = make_kreuzberg_document(doc_id="k-001")
    plain_doc = Document(
        text="Plain",
        id_="plain-001",
        metadata={"_kreuzberg_elements": "not-a-list", "file_path": "/tmp/test.txt"},
    )
    parser = KreuzbergNodeParser()

    nodes = parser.get_nodes_from_documents([kreuzberg_doc, plain_doc])

    kreuzberg_children = [n for n in nodes if n.source_node and n.source_node.node_id == "k-001"]
    assert len(kreuzberg_children) == 3
    for child in kreuzberg_children:
        assert "_kreuzberg_elements" not in child.metadata

    passthrough = [n for n in nodes if n.node_id == "plain-001"]
    assert len(passthrough) == 1
    assert "_kreuzberg_elements" in passthrough[0].metadata


def test_custom_id_func_respected() -> None:
    doc = make_kreuzberg_document()
    parser = KreuzbergNodeParser(
        id_func=lambda i, doc: f"custom-{doc.node_id}-{i}",
    )

    nodes = parser.get_nodes_from_documents([doc])

    assert nodes[0].id_ == "custom-doc-001-0"
    assert nodes[1].id_ == "custom-doc-001-1"
    assert nodes[2].id_ == "custom-doc-001-2"


def test_metadata_propagation_and_exclusion_keys() -> None:
    doc = make_kreuzberg_document()
    parser = KreuzbergNodeParser()

    nodes = parser.get_nodes_from_documents([doc])

    for node in nodes:
        assert "element_type" in node.excluded_embed_metadata_keys
        assert "page_number" in node.excluded_embed_metadata_keys
        assert "element_index" in node.excluded_embed_metadata_keys
        assert "element_type" not in node.excluded_llm_metadata_keys
        assert "page_number" not in node.excluded_llm_metadata_keys
        assert "element_index" not in node.excluded_llm_metadata_keys
        assert node.metadata.get("file_path") == "/tmp/test.pdf"
        assert node.metadata.get("mime_type") == "application/pdf"


def test_mixed_batch_splits_kreuzberg_passes_through_others() -> None:
    kreuzberg_doc = make_kreuzberg_document(doc_id="k-001")
    plain_doc = Document(text="Plain text", id_="plain-001", metadata={"file_path": "/tmp/plain.txt"})
    parser = KreuzbergNodeParser()

    nodes = parser.get_nodes_from_documents([kreuzberg_doc, plain_doc])

    assert len(nodes) == 4
    assert nodes[-1].text == "Plain text"
    assert nodes[-1].id_ == "plain-001"


def test_call_contract_sequence_in_sequence_out() -> None:
    doc = make_kreuzberg_document()
    parser = KreuzbergNodeParser()

    nodes = parser([doc])

    assert isinstance(nodes, list)
    assert len(nodes) == 3
    assert all(isinstance(n, BaseNode) for n in nodes)
