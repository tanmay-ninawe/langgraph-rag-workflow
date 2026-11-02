"""
Run once to build the FAISS vector index from sample_docs/*.txt
Usage: python ingest.py
"""

import glob
import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Force find the .env file using its absolute path relative to this script file
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")
load_dotenv(dotenv_path=env_path)

DOCS_DIR = os.path.join(script_dir, "Sample_documents")
INDEX_DIR = os.path.join(script_dir, "faiss_index")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Loading the file and making the file ready for chunking.
def load_and_split() -> list[Document]:
    """Load .txt files and split into ~500-char chunks with overlap."""
    chunks = []
    for path in glob.glob(os.path.join(DOCS_DIR, "*.txt")):
        with open(path, "r", encoding="utf-8") as f: #keeping the file in Ready mode 'r'
            text = f.read()

        source = os.path.basename(path)
        # Split on double-newline (paragraph boundary)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= CHUNK_SIZE:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(Document(page_content=current, metadata={"source": source}))
                # If single paragraph is already > CHUNK_SIZE, keep it as-is
                current = para

        if current:
            chunks.append(Document(page_content=current, metadata={"source": source}))

    return chunks


def main():
    # Verify the API key exists before attempting to connect to Google's servers
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: API Key missing!")
        print(f"👉 Please make sure a file named exactly '.env' exists in: {script_dir}")
        print("   Inside it, add your token: GEMINI_API_KEY=your_actual_api_key")
        return

    print("Loading and splitting documents...")
    chunks = load_and_split()
    print(f"  Split into {len(chunks)} chunks")

    if not chunks:
        print("❌ Ingestion halted: 0 text chunks were loaded from your folder.")
        return

    print("Embedding and building FAISS index...")
    # Fixed: Bound the api_key parameter explicitly to ensure it validates
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )
    
    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(INDEX_DIR)
        print(f"Done! Index saved to {INDEX_DIR}/")
        print(f"  Total chunks indexed: {len(chunks)}")
    except Exception as e:
        print(f"❌ FAISS Ingestion failed: {e}")


if __name__ == "__main__":
    main()
