import json

import pandas as pd

from .ai import get_answer_from_ai
from .constants import AvailableEmbeddingModels, AvailableModels
from .questions import ProcessedQuestion, json_to_dict


def process_lease(
    file_name: str,
    namespace: str,
    question_for_ai: ProcessedQuestion,
    question_id: int,
    answers_df: pd.DataFrame,
    index_name: str,
    model: AvailableModels = "gpt-3.5-turbo",
    embedding_model: AvailableEmbeddingModels = "text-embedding-ada-002",
    use_structured_outputs=False,
):
    answer_from_ai = get_answer_from_ai(
        json.dumps(question_for_ai),
        namespace,
        model=model,
        embedding_model=embedding_model,
        supports_structured_output=use_structured_outputs,
        index_name=index_name,
    )
    question_id_str = f"{question_id}"
    answers_for_lease_df = answers_df[answers_df["Lease"] == file_name][question_id]

    if use_structured_outputs and model != "gpt-3.5-turbo":
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
