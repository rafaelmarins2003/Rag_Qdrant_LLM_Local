from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from qdrant_client.http.models import Distance, VectorParams
from fastapi import FastAPI, UploadFile, File
import uvicorn, qdrant_client, uuid
from pathlib import Path
from pydantic import BaseModel

from memoria import SimpleChatHistory, MemoryConfig, ChatMemory

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
     "Se a informação não estiver lá, responda: «Não sei com base no contexto.»\n"
     "Escreva em Português‑BR, de forma concisa e direta."),
    # 2 · contexto dos documentos --------------------------------------------
    ("system", "<<CTX>>\n{context}\n<</CTX>>"),
    # 3 · instrução para follow‑ups ------------------------------------------
    ("system",
     "Se a pergunta do usuário for um pedido de esclarecimento, explicação melhor, ou aprofundamento sobre\n"
     " tente ajuda-lo trazendo outras formas de explicar aquilo que você explicou anteriormente,\n"
     " evitando dar a mesma resposta."),
    # 4 · user ----------------------------------------------------------------
    ("user",
     "{question}\n\n"
     "• Use listas numeradas quando fizer sentido.\n"
     "• Não acrescente explicações longas demais nem divague em tópicos não relacionados.\n"
     )
])

memory = ChatMemory(
    history=SimpleChatHistory(),
    config=MemoryConfig(max_messages=6)
)

@app.post("/chat")
async def chat(req: ChatRequest):

    user_msg = HumanMessage(content=req.question)

    # Retrieve top‑k chunks
    retriever = vs.as_retriever(
        search_type="similarity",  # similarity: similaridade, mmr: diversidade
        search_kwargs={"k": 6}
    )

    # 2. na rota /chat
    docs = retriever.invoke(req.question)
    context = "\n\n".join(d.page_content for d in docs)

    mem_msgs = memory.load_memory()
    memoria_str = "\n".join(
        f"{'Usuário' if isinstance(m, HumanMessage) else 'Assistente'}: {m.content}"
        for m in mem_msgs
    )

    if memoria_str:
        full_context = memoria_str + "\n\n" + context
    else:
        full_context = context

    chain = prompt_template | llm
    raw_response = chain.invoke({
        "question": req.question,
        "context": full_context
    })

    if isinstance(raw_response, BaseMessage):
        answer_text = raw_response.content
    elif isinstance(raw_response, list):
        # se por acaso vier uma lista de BaseMessage
        answer_text = "\n".join(
            m.content if isinstance(m, BaseMessage) else str(m)
            for m in raw_response
        )
    else:
        answer_text = str(raw_response)

    ai_msg = AIMessage(content=answer_text)
    memory.save_context(user_msg, ai_msg)

    return {"answer": answer_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

