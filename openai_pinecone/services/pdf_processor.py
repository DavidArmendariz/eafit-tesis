from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.document_loaders import BaseLoader


class Ingestor:
    def __init__(
        self,
        loader: BaseLoader,
        index_name: str,
        namespace: str,
        embeddings: OpenAIEmbeddings = OpenAIEmbeddings(),
        splitter=RecursiveCharacterTextSplitter(),
    ):
        self.index_name = index_name
        self.namespace = namespace
        self.embeddings = embeddings
        self.loader = loader
        self.splitter = splitter

    def load(self):
        self.documents = self.loader.load()

    def split(self):
        self.splits = self.splitter.split_documents(self.documents)

    def store(self):
        documents: list[Document] = []
        ids = []

        for index, document in enumerate(self.splits):
            ids.append(f"{self.namespace}#chunk{index}")
            metadata = document.metadata
            documents.append(
                Document(
                    ids=ids[index],
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

        PineconeVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            index_name=self.index_name,
            namespace=self.namespace,
            ids=ids,
        )
