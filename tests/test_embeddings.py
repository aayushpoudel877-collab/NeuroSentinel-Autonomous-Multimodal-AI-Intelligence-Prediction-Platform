from app.models.embeddings import HashingTextEmbedder
def test_embedding_shape():
    vector=HashingTextEmbedder(32).encode('multimodal intelligence platform')
    assert len(vector)==32
