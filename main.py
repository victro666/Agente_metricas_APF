import io
import json # Nova importação
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse, Response # Adicionado Response
from pypdf import PdfReader
from dotenv import load_dotenv

from schemas import RequisitoTextoInput, RelatorioContagemResponse
from agent import CopilotoAPFAgent

load_dotenv()

estado_app = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Inicializando Copiloto de Métricas APF (Google AI Studio + ChromaDB)...")
    estado_app["agente"] = CopilotoAPFAgent()
    print("✅ Agente pronto para receber requisições!")
    yield
    estado_app.clear()

app = FastAPI(
    title="Agente Contagem Estimativa APF",
    description="API Agentic RAG para extração e classificação de CFBs (IFPUG CPM 4.3.1 / NESMA Estimativa).",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def pagina_inicial():
    return FileResponse("index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "online", "motor": "Google Gemini 3.5 Flash-Lite", "rag": "ChromaDB Local"}

@app.post("/api/v1/extrair-texto", response_model=RelatorioContagemResponse)
def extrair_de_texto(payload: RequisitoTextoInput):
    agente: CopilotoAPFAgent = estado_app.get("agente")
    if not agente:
        raise HTTPException(status_code=503, detail="Agente não inicializado.")
    return agente.processar_requisitos(payload.nome_projeto, payload.texto_requisitos)

@app.post("/api/v1/extrair-arquivo", response_model=RelatorioContagemResponse)
async def extrair_de_arquivo(arquivo: UploadFile = File(...)):
    agente: CopilotoAPFAgent = estado_app.get("agente")
    if not agente:
        raise HTTPException(status_code=503, detail="Agente não inicializado.")

    nome_arquivo = arquivo.filename or "documento.txt"
    extensao = nome_arquivo.lower().split(".")[-1]
    conteudo_bytes = await arquivo.read()

    if extensao == "pdf":
        leitor_pdf = PdfReader(io.BytesIO(conteudo_bytes))
        texto_extraido = "\n".join([pagina.extract_text() or "" for pagina in leitor_pdf.pages])
    elif extensao == "txt":
        texto_extraido = conteudo_bytes.decode("utf-8", errors="ignore")
    else:
        raise HTTPException(status_code=400, detail="Formato não suportado. Envie um arquivo .pdf ou .txt.")

    if len(texto_extraido.strip()) < 20:
        raise HTTPException(status_code=422, detail="O documento enviado não possui texto legível suficiente.")

    return agente.processar_requisitos(nome_arquivo, texto_extraido)

# --- NOVA ROTA DE EXPORTAÇÃO ---
@app.post("/api/v1/exportar-json")
def exportar_json(payload: RelatorioContagemResponse):
    json_str = payload.model_dump_json(indent=4)
    return Response(
        content=json_str,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=extracao_{payload.nome_projeto}.json"}
    )