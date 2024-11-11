import pandas as pd

from notebooks.utils.ai_v2 import get_answer_from_ai_v2


def process_lease_v2(
    file_name: str,
    namespace: str,
    question_id: int,
    answers_df: pd.DataFrame,
    index_name: str,
    lease_number: int,
    template_string: str | None = None,
    retriever_question: str | None = None,
    use_llm_filter=False,
    use_listwise_rerank=False,
    use_embeddings_filter=False,
    lessor_question=True,
    date_question=False,
    boolean_question=False,
    temperature=0.0,
    chunks_filename: str | None = None,
):
    answer_from_ai = get_answer_from_ai_v2(
        namespace,
        index_name=index_name,
        template_string=template_string,
        retriever_question=retriever_question,
        use_llm_filter=use_llm_filter,
        lessor_question=lessor_question,
        date_question=date_question,
        boolean_question=boolean_question,
        temperature=temperature,
        use_listwise_rerank=use_listwise_rerank,
        use_embeddings_filter=use_embeddings_filter,
        chunks_filename=chunks_filename,
        lease_number=lease_number,
    )
    answers_for_lease_df = answers_df[answers_df["Lease"] == file_name][
        question_id
    ].iloc[0]

    return {
        "answer": answer_from_ai,
        "real_answer_unprocessed": answers_for_lease_df,
    }
