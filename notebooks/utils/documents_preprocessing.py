from openai_pinecone.services.upload_to_s3 import upload_to_s3

from .constants import AvailableEmbeddingModels
from .storage import store_pdf_in_vector_db


class DocumentsPreprocessing:
    def __init__(
        self,
        index_name: str,
        embedding_model: AvailableEmbeddingModels,
    ):
        self.numbers_list = [str(i).zfill(3) for i in range(1, 101)]
        self.index_name = index_name
        self.embedding_model: AvailableEmbeddingModels = embedding_model

    def store_in_s3(self):
        for number in self.numbers_list:
            file_name = f"lease{number}"
            local_file = f"../data/asc_842/lease_agreements/{file_name}.pdf"
            file_key = f"eafit/{file_name}.pdf"
            upload_to_s3(local_file, file_key)

    def store_in_vector_db(self):
        for number in self.numbers_list:
            file_name = f"lease{number}"
            file_key = f"eafit/{file_name}.pdf"
            namespace = f"eafit_{file_name}"
            store_pdf_in_vector_db(
                file_key,
                namespace,
                embedding_model=self.embedding_model,
                index_name=self.index_name,
            )
