import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

CHROMA_PATH = "./chroma_db"
ARQUIVO_REGRAS = "./regras_CPM.txt"

def construir_banco_local():
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("Configure a variável GOOGLE_API_KEY no seu .env")

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    docs_regras = TextLoader(ARQUIVO_REGRAS, encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits_regras = splitter.split_documents(docs_regras)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    Chroma.from_documents(
        documents=splits_regras,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="cpm_ifpug_rules"
    )
    print("✅ Novo 'chroma_db/' gerado via Google Embeddings! Faça o commit desta pasta.")

if __name__ == "__main__":
    construir_banco_local()