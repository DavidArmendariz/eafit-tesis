import json
import os

import pandas as pd

from notebooks.utils.ai import get_answer_from_ai
from notebooks.utils.questions import ProcessedQuestion, json_to_dict
from notebooks.utils.storage import store_pdf_in_vector_db
from openai_pinecone.services.store_pdf import upload_to_s3

from .constants import AvailableEmbeddingModels, AvailableModels


def process_lease(
    local_file: str,
    file_name: str,
    file_key: str,
    namespace: str,
    question_for_ai: ProcessedQuestion,
    question_id: int,
    answers_df: pd.DataFrame,
    model: AvailableModels = "gpt-3.5-turbo",
    embedding_model: AvailableEmbeddingModels = "text-embedding-ada-002",
    already_stored=False,
    already_vectorized=False,
    force_structured_output=False,
    index_name=os.getenv("INDEX_NAME", ""),
):
    if not already_stored:
        upload_to_s3(local_file, file_key)

    if not already_vectorized:
        store_pdf_in_vector_db(
            file_key, namespace, embedding_model=embedding_model, index_name=index_name
        )
    supports_structured_output = (
        model in ["gpt-4o-2024-08-06", "gpt-4o-mini"]
    ) or force_structured_output
    answer_from_ai = get_answer_from_ai(
        json.dumps(question_for_ai),
        namespace,
        model=model,
        embedding_model=embedding_model,
        supports_structured_output=supports_structured_output,
        use_json_object=force_structured_output,
        index_name=index_name,
    )
    question_id_str = f"{question_id}"
    answers_for_lease_df = answers_df[answers_df["Lease"] == file_name][question_id]
    if model in ["gpt-4o-2024-08-06", "gpt-4o-mini"]:
        return {
            "answer": answer_from_ai,
            "real_answer_unprocessed": answers_for_lease_df,
        }
    answer_as_dict = json_to_dict(answer_from_ai)
    if not answer_as_dict:
        print("No answer found")
        return {"answer": None, "real_answer_unprocessed": answers_for_lease_df}
    answer = answer_as_dict[question_id_str]["answer"]
    return {"answer": answer, "real_answer_unprocessed": answers_for_lease_df}
