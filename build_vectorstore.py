import os
import shutil
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

class OnnxMiniLMEmbeddings(Embeddings):
    """Wrapper leve para o all-MiniLM-L6-v2 via ONNX nativo do ChromaDB (Sem PyTorch)."""
    def __init__(self):
        self._ef = DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [list(map(float, vec)) for vec in self._ef(texts)]

    def embed_query(self, text: str) -> List[float]:
        return list(map(float, self._ef([text])[0]))

CHROMA_PATH = "./chroma_db"
ARQUIVO_REGRAS = "./regras_CPM.txt"

def construir_banco_local():
    if not os.path.exists(ARQUIVO_REGRAS):
        raise FileNotFoundError(f"Arquivo '{ARQUIVO_REGRAS}' não encontrado na raiz do projeto.")

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    print("Lendo regras do Manual CPM...")
    docs_regras = TextLoader(ARQUIVO_REGRAS, encoding="utf-8").load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n## ", "\n* ", "\n\n", "\n", " "]
    )
    splits_regras = splitter.split_documents(docs_regras)

    print(f"Gerando banco vetorial persistente em '{CHROMA_PATH}' ({len(splits_regras)} blocos)...")
    embeddings = OnnxMiniLMEmbeddings()
    Chroma.from_documents(
        documents=splits_regras,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="cpm_ifpug_rules"
    )
    print("✅ Banco 'chroma_db/' criado com sucesso! Faça o commit desta pasta no GitHub.")

if __name__ == "__main__":
    construir_banco_local()