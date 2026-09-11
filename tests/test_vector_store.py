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


# -- sync_documents() / Indexing API integration --------------------------


def test_sync_documents_adds_new_content(sample_documents, fake_embedding_service, tmp_path):
    store = VectorStore(embedding_service=fake_embedding_service, persist_dir=tmp_path)
    result = store.sync_documents(sample_documents)

    assert result["num_added"] == len(sample_documents)
    assert store.exists_on_disk()
    assert all(store.is_indexed(doc.metadata["source"]) for doc in sample_documents)


def test_sync_documents_skips_unchanged_content(sample_documents, fake_embedding_service, tmp_path):
    store = VectorStore(embedding_service=fake_embedding_service, persist_dir=tmp_path)
    store.sync_documents(sample_documents)

    result = store.sync_documents(sample_documents)

    assert result["num_added"] == 0
    assert result["num_skipped"] == len(sample_documents)


def test_sync_documents_removes_source_no_longer_present(
    sample_documents, fake_embedding_service, tmp_path
):
    store = VectorStore(embedding_service=fake_embedding_service, persist_dir=tmp_path)
    store.sync_documents(sample_documents)
    removed_source = sample_documents[0].metadata["source"]

    remaining = [doc for doc in sample_documents if doc.metadata["source"] != removed_source]
    result = store.sync_documents(remaining)

    assert result["num_deleted"] == 1
    assert store.is_indexed(removed_source) is False
    assert all(store.is_indexed(doc.metadata["source"]) for doc in remaining)


def test_sync_documents_to_empty_list_forgets_the_index(
    sample_documents, fake_embedding_service, tmp_path
):
    store = VectorStore(embedding_service=fake_embedding_service, persist_dir=tmp_path)
    store.sync_documents(sample_documents)
    assert store.exists_on_disk()

    store.sync_documents([])

    # Emptied out completely -- should look identical to "never built",
    # not a zero-length-but-present index.
    assert store.exists_on_disk() is False
    assert store.is_indexed(sample_documents[0].metadata["source"]) is False


def test_is_indexed_false_before_any_sync(fake_embedding_service, tmp_path):
    store = VectorStore(embedding_service=fake_embedding_service, persist_dir=tmp_path)
    assert store.is_indexed("never_uploaded.pdf") is False
