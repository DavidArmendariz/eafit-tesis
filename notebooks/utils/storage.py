import os

from dotenv import load_dotenv
from langchain_community.document_loaders import S3FileLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openai_pinecone.services.pdf_processor import Ingestor

from .constants import AvailableEmbeddingModels

load_dotenv()

bucket_name = os.getenv("BUCKET_NAME", "")


def store_pdf_in_vector_db(
    file_key: str,
    namespace: str,
    index_name: str,
    embedding_model: AvailableEmbeddingModels,
):
    try:
        ingestor = Ingestor(
            loader=S3FileLoader(bucket_name, file_key),
            namespace=namespace,
            splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200),
            embeddings=OpenAIEmbeddings(model=embedding_model),
            index_name=index_name,
        )
        ingestor.load()
        ingestor.split()
        ingestor.store()
        print(f"Stored {file_key} in vector database with namespace {namespace}")
    except Exception as e:
        print(f"Error processing {file_key}: {e}")
        raise e
