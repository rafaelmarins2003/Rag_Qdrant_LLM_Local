from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI, UploadFile, File
import uvicorn, qdrant_client, uuid
from pathlib import Path
from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str

UPLOAD_DIR = Path("./tmp")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- LLM & embedding nodes ---------------------------------------------------
llm      = ChatOllama(model="qwen2.5:7b", temperature=0.3)
embedder = OllamaEmbeddings(model="nomic-embed-text")

# --- Vector DB ---------------------------------------------------------------
client = qdrant_client.QdrantClient(url="http://localhost:6333")
collection = "rag_uploads"
if collection not in [c.name for c in client.get_collections().collections]:
    dim = len(embedder.embed_query("test"))
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    _ = QdrantVectorStore(
        client = client,
        collection_name=collection,
        embedding=embedder,
    )
vs = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embedder
     )

# --- FastAPI server ----------------------------------------------------------
app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # -- save ---------------------------------------------------------------
    path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    path.write_bytes(await file.read())

    # -- read & chunk -------------------------------------------------------
    loader    = PyPDFLoader(str(path))
    pages     = loader.load()                                 # -> list[Document]
    splitter  = RecursiveCharacterTextSplitter(
                    chunk_size=1200, chunk_overlap=80,
                    separators=["\n\n", "\n", " ", ""]
    )
    chunks    = splitter.split_documents(pages)

    # -- embed & upsert -----------------------------------------------------
    vs.add_documents(chunks)
    return {"chunks": len(chunks)}

prompt_template = ChatPromptTemplate.from_messages([
    # 1 · system -------------------------------------------------------------
    ("system",
     "Você é um assistente especializado em FINANÇAS que usa RAG.\n"
     "Responda APENAS com base no CONTEXTO fornecido entre <<CTX>> … <</CTX>>;\n"
     "se a informação não estiver ali, responda: «Não sei com base no contexto.»\n"
     "Escreva em Português‑BR, em formato conciso e direto."),
    # 2 · context placeholder ------------------------------------------------
    ("system", "<<CTX>>\n{context}\n<</CTX>>"),
    # 3 · user ----------------------------------------------------------------
    ("user",
     "{question}\n\n"
     "• Use listas numeradas quando fizer sentido.\n"
     "• Não acrescente explicações longas e genéricas.\n"
     )
])

@app.post("/chat")
async def chat(req: ChatRequest):
    # Retrieve top‑k chunks
    retriever = vs.as_retriever(
        search_type="similarity",  # similarity: similaridade, mmr: diversidade
        search_kwargs={"k": 6}
    )

    # 2. na rota /chat
    docs = retriever.invoke(req.question)
    context = "\n\n".join(d.page_content for d in docs)
    chain = prompt_template | llm
    print({"question": req.question, "context": context})
    answer = chain.invoke({"question": req.question, "context": context})
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


