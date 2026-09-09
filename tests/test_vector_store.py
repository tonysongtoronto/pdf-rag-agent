from src.vector_store import VectorStore


def test_build_index_from_documents(sample_documents, fake_embedding_service, tmp_path):
    store = VectorStore(embedding_service=fake_embedding_service, persist_dir=tmp_path)
    store.build_from_documents(sample_documents)
    assert store.store is not None


def test_add_documents_to_existing_index(fake_vector_store, sample_documents):
    extra = [sample_documents[0]]
    fake_vector_store.add_documents(extra)
    results = fake_vector_store.similarity_search("revenue", top_k=10)
    assert len(results) >= len(sample_documents)


def test_save_and_load_index(fake_vector_store, tmp_path):
    fake_vector_store.save()
    assert fake_vector_store.exists_on_disk()

    reloaded = VectorStore(
        embedding_service=fake_vector_store.embedding_service,
        persist_dir=tmp_path,
        index_name=fake_vector_store.index_name,
    )
    reloaded.load()
    results = reloaded.similarity_search("revenue", top_k=2)
    assert len(results) > 0


def test_similarity_search_returns_documents(fake_vector_store):
    results = fake_vector_store.similarity_search("battery warranty", top_k=2)
    assert len(results) > 0
    assert all(hasattr(doc, "page_content") for doc in results)


def test_store_property_raises_when_nothing_built_or_saved(tmp_path, fake_embedding_service):
    store = VectorStore(embedding_service=fake_embedding_service, persist_dir=tmp_path)
    import pytest

    with pytest.raises(RuntimeError):
        _ = store.store
