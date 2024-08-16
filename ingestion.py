import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()
from langchain_pinecone import Pinecone


# Function to process a single PDF file
def process_pdf(file_path, index_name):
    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Upload to Pinecone
    vectorstore_from_docs = Pinecone.from_documents(
        texts,
        index_name=index_name,
        embedding=embeddings
    )


# Main function to process all PDFs in a directory
def ingest_pdfs(directory_path, index_name):
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing {filename}...")
            process_pdf(file_path, index_name)
    print("All PDFs processed and ingested into Pinecone.")


# Usage
ingest_pdfs("data", "documentation-helper")
