def test_retriever_returns_results(fake_retriever):
    results = fake_retriever.invoke("What is the company's revenue?")
    assert len(results) > 0


def test_retriever_respects_top_k(fake_retriever):
    results = fake_retriever.invoke("revenue", top_k=1)
    assert len(results) <= 1


def test_retriever_results_have_content_and_metadata(fake_retriever):
    results = fake_retriever.invoke("warranty battery")
    for doc in results:
        assert doc.page_content.strip() != ""
        assert "source" in doc.metadata
        assert "page" in doc.metadata
