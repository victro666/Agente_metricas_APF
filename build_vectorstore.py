import os
import shutil
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
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

    # Vetorização via Google (100% Gratuito)
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=google_api_key,
        model="gemini-embedding-2" # Modelo mais recente e otimizado do Google
    )
    
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="cpm_ifpug_rules"
    )

    tamanho_lote = 80
    for i in range(0, len(splits_regras), tamanho_lote):
        lote = splits_regras[i : i + tamanho_lote]
        print(f"Vetorizando lote {i} até {i + len(lote)} de {len(splits_regras)}...")
        db.add_documents(lote)
        
        if i + tamanho_lote < len(splits_regras):
            print("Aguardando 60 segundos para respeitar a cota de embeddings do Google...")
            time.sleep(60)
            
    print("✅ Novo 'chroma_db/' gerado com sucesso via Google! Faça o commit desta pasta.")

if __name__ == "__main__":
    construir_banco_local()