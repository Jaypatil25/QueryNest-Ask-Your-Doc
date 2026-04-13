# import os
# import shutil
# from pathlib import Path

# from dotenv import load_dotenv
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel

# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_groq import ChatGroq
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA

# load_dotenv()

# app = FastAPI(title="RAG Application")

# # --------------- state ---------------
# UPLOAD_DIR = Path("uploads")
# UPLOAD_DIR.mkdir(exist_ok=True)

# vector_store = None  # will hold the FAISS index after upload


# # --------------- helpers ---------------

# def load_document(file_path: str):
#     """Load a PDF or text file and return LangChain Documents."""
#     # TODO 


# def build_vector_store(documents):
#     """Split documents into chunks and build a FAISS vector store."""
#     # TODO 


# def get_qa_chain(store):
#     """Create a RetrievalQA chain from the vector store."""
#     # TODO 

# # --------------- routes ---------------

# @app.get("/", response_class=HTMLResponse)
# async def home():
#     # TODO 


# class QueryRequest(BaseModel):
#     question: str


# @app.post("/upload")
# async def upload_document(file: UploadFile = File(...)):
#     # TODO


# @app.post("/query")
# async def query_document(req: QueryRequest):
#     #TODO 






import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()

app = FastAPI(title="QueryNest")
app.mount("/public", StaticFiles(directory="public"), name="public")

# --------------- state ---------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

vector_store = None

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


# --------------- helpers ---------------

def load_document(file_path: str):
    """Load a PDF or text file and return LangChain Documents."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Only .pdf and .txt files are supported.")
    
    return loader.load()


def build_vector_store(documents):
    """Split documents into chunks and build a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks = splitter.split_documents(documents)
    
    store = FAISS.from_documents(chunks, get_embeddings())
    return store


def get_qa_chain(store):
    """Create a RetrievalQA chain from the vector store."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return chain


# --------------- routes ---------------

@app.get("/", response_class=HTMLResponse)
async def home():
    return Path("static/index.html").read_text()


class QueryRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".pdf", ".txt"]:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    safe_name = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_name

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        documents = load_document(str(file_path))
        vector_store = build_vector_store(documents)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": f"'{safe_name}' uploaded successfully.",
        "pages": len(documents)
    }


@app.post("/query")
async def query_document(req: QueryRequest):
    global vector_store

    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Please upload a document first."
        )

    chain = get_qa_chain(vector_store)
    result = chain.invoke({"query": req.question})

    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "content": doc.page_content[:300],
            "metadata": doc.metadata
        })

    return {
        "answer": result["result"],
        "sources": sources
        }