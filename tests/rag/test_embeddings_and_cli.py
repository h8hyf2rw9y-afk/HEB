import io
import json
import os

import pytest

from app.rag import cli
from app.rag import embeddings as emb
from app.rag.embeddings import EmbeddingError, OllamaEmbedder, cosine_similarities


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_ollama_embedder_parses_response_and_usage(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data)
        n = len(sent["body"]["input"])
        return _Resp(json.dumps({"embeddings": [[1.0, 0.0]] * n, "prompt_eval_count": 7}).encode())

    monkeypatch.setattr(emb.urllib.request, "urlopen", fake_urlopen)
    e = OllamaEmbedder(model="bge-m3", base_url="http://ollama:11434/")
    vecs, usage = e.embed(["a", "b"])
    assert sent["url"] == "http://ollama:11434/api/embed"
    assert sent["body"] == {"model": "bge-m3", "input": ["a", "b"]}
    assert vecs == [[1.0, 0.0], [1.0, 0.0]]
    assert usage.calls == 1 and usage.input_tokens == 7


def test_ollama_embedder_rejects_malformed_response(monkeypatch):
    monkeypatch.setattr(emb.urllib.request, "urlopen", lambda req, timeout: _Resp(b'{"error":"model not found"}'))
    with pytest.raises(EmbeddingError):
        OllamaEmbedder().embed(["a"])


def test_ollama_unreachable_is_reported():
    e = OllamaEmbedder(base_url="http://127.0.0.1:9", timeout_s=1)
    assert e.is_available() is False


def test_cosine():
    assert cosine_similarities([1, 0], [[1, 0], [0, 1], [0, 0]]) == [1.0, 0.0, 0.0]


def test_cli_end_to_end(tmp_path, capsys):
    pdfs, idx = tmp_path / "pdfs", tmp_path / "idx"
    assert cli.main(["synthetic", "--out", str(pdfs)]) == 0
    assert cli.main(["build", "--pdf-dir", str(pdfs), "--index-dir", str(idx), "--embeddings", "off"]) == 0
    capsys.readouterr()
    assert cli.main(["query", "¿Quién puede operar montacargas?", "--index-dir", str(idx), "--embeddings", "off"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "answered"
    assert out["evidence"][0]["page"] == 1


def test_cli_build_without_pdfs(tmp_path):
    assert cli.main(["build", "--pdf-dir", str(tmp_path), "--index-dir", str(tmp_path / "i"), "--embeddings", "off"]) == 1


@pytest.mark.skipif(os.getenv("HEB_RAG_OLLAMA_TESTS") != "1", reason="requiere Ollama local (HEB_RAG_OLLAMA_TESTS=1)")
def test_real_ollama_hybrid(pdf_paths):
    from app.rag.config import RagSettings
    from app.rag.index import PolicyIndex
    from app.rag.service import PolicyRetriever

    s = RagSettings.from_env()
    e = OllamaEmbedder(s.embed_model, s.ollama_base_url)
    index = PolicyIndex.build(pdf_paths, RagSettings(embeddings="required"), e)
    ans = PolicyRetriever(index, s, e).answer("¿Cuántos días tengo para devolver un producto?")
    assert ans.metrics.retrieval_mode == "hybrid"
    assert ans.status == "answered"
