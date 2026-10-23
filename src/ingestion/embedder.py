from config import settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from google import genai
from google.genai import types
from langchain_core.documents import Document
import os
from typing import List
from langchain_community.vectorstores import Chroma


import hashlib, json, time
from pathlib import Path
from typing import List
from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma  # pip install langchain-chroma


class GoogleGeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str, cache_dir: str = "./embedding_cache"):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    # --- cache ---
    def _path(self, text: str, task: str) -> Path:
        key = hashlib.sha256(f"{self.model_name}|{task}|{text}".encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _get(self, text, task):
        p = self._path(text, task)
        return json.loads(p.read_text()) if p.exists() else None

    def _set(self, text, task, vec):
        self._path(text, task).write_text(json.dumps(vec))

    # --- appel API avec retry ---
    def _embed(self, batch: List[str], task: str) -> List[List[float]]:
        for attempt in range(5):
            try:
                r = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task),
                )
                return [e.values for e in r.embeddings]
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    time.sleep(2 ** attempt * 5)
                else:
                    raise
        raise RuntimeError("Quota Gemini dépassé")

    def _embed_with_cache(self, texts: List[str], task: str, batch_size=50):
        results = [self._get(t, task) for t in texts]
        missing = [i for i, v in enumerate(results) if v is None]
        for i in range(0, len(missing), batch_size):
            idx = missing[i:i + batch_size]
            vecs = self._embed([texts[j] for j in idx], task)
            for j, v in zip(idx, vecs):
                results[j] = v
                self._set(texts[j], task, v)
        return results

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_with_cache(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, query: str) -> List[float]:
        return self._embed_with_cache([query], "RETRIEVAL_QUERY")[0]
    
import shutil
from config import settings

def build_embeddings(
    api_key: str | None = None,
    model_name: str | None = None,
    cache_dir: str = "./embedding_cache",
) -> GoogleGeminiEmbeddings:
    return GoogleGeminiEmbeddings(
        api_key=api_key or settings.google_api_key,
        model_name=model_name or settings.embedding_model,
        cache_dir=cache_dir,
    )
def build_vectorstore(chunks, embeddings, persist_directory="./chroma_db", rebuild=False):
    if rebuild:
        shutil.rmtree(persist_directory, ignore_errors=True)
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)