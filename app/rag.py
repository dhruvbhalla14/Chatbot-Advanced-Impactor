import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredPowerPointLoader,
    PyPDFLoader,
    Docx2txtLoader
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")

# ---------------- LOAD DOCUMENTS ---------------- #

def load_documents():
    loaders = [
        PyPDFLoader(str(DATA_DIR / "Price plan.pdf")),
        PyPDFLoader(str(DATA_DIR / "PLANS AND FEATURES PHASE 1_removed.pdf")),
        Docx2txtLoader(str(DATA_DIR / "01_Privacy_Policy.docx")),
        Docx2txtLoader(str(DATA_DIR / "02_Terms_of_Use_EULA.docx")),
        Docx2txtLoader(str(DATA_DIR / "03_Refund_Cancellation_Policy.docx")),
        Docx2txtLoader(str(DATA_DIR / "04_Cookie_Policy.docx")),
        Docx2txtLoader(str(DATA_DIR / "05_Coaches_Academy_Terms.docx")),
        Docx2txtLoader(str(DATA_DIR / "06_Enterprise_B2B_Terms.docx")),
        Docx2txtLoader(str(DATA_DIR / "07_AI_Output_Disclaimer.docx")),
        Docx2txtLoader(str(DATA_DIR / "08_Minor_Parental_Consent.docx")),
        Docx2txtLoader(str(DATA_DIR / "Advanced Impactor MVP+Next Phase Proposal V1.0.docx")),
        UnstructuredPowerPointLoader(str(DATA_DIR / "AdvancedImpactor_Deck_V7 (1).pptx")),
    ]
    
    docs = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
            print(f"SUCCESS Loaded - {loader.file_path}")
        except Exception as e:
            print(f"ERROR Skipping - {loader.file_path} -> {e}")
            continue
    
    return docs
# ---------------- CLEANING ---------------- #

def clean_documents(documents):
    cleaned_docs = []

    for doc in documents:
        text = doc.page_content
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()

        if len(text) > 30:
            cleaned_docs.append(Document(
                page_content=text,
                metadata=doc.metadata
            ))

    return cleaned_docs


# ---------------- CHUNKING ---------------- #

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    return splitter.split_documents(docs)


# ---------------- FILTERING ---------------- #

JUNK_PATTERNS = [
    r'click here',
    r'figma',
    r'confidential',
    r'competitor analysis',
]

def is_junk(text):
    t = text.lower()
    for pattern in JUNK_PATTERNS:
        if re.search(pattern, t):
            return True
    return len(text.strip()) < 60


def filter_chunks(chunks):
    return [doc for doc in chunks if not is_junk(doc.page_content)]


# ---------------- VECTOR STORE ---------------- #


def create_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
    batch_size = 50
    vectorstore = None
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        print(f"Processed batch {i} to {i+len(batch)} out of {len(chunks)} chunks...")
        if i + batch_size < len(chunks):
            time.sleep(30) # Wait to avoid Google API free tier rate limits
            
    vectorstore.save_local(str(VECTORSTORE_DIR))
    return vectorstore


def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

# ---------------- RAG CHAIN ---------------- #

def create_rag_chain(vectorstore):

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    prompt = PromptTemplate(
        template="""You are a helpful assistant for Advanced Impactor.

Answer the user's question using ONLY the information provided in the context below.
If the answer is not found in the context, say: "I don't have enough information to answer that."
Do NOT make up or assume any information.
Be concise, clear, and professional in your response.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
