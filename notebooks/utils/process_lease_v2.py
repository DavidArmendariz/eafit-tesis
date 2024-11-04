import pandas as pd

from notebooks.utils.ai_v2 import get_answer_from_ai_v2


def process_lease_v2(
    file_name: str,
    namespace: str,
    question_id: int,
    answers_df: pd.DataFrame,
    index_name: str,
    template_string: str | None = None,
    retriever_question: str | None = None,
    use_llm_filter=False,
    lessor_question=True,
    date_question=False,
    boolean_question=False,
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
    )
    answers_for_lease_df = answers_df[answers_df["Lease"] == file_name][
        question_id
    ].iloc[0]

    return {
        "answer": answer_from_ai,
        "real_answer_unprocessed": answers_for_lease_df,
    }
