from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from openai_pinecone.services.question_answerer_v2 import QuestionAnswererV2
from openai_pinecone.utils.asc_842 import asc_842_prompt_template_structured


def get_answer_from_ai_v2(
    namespace: str,
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
    try:
        prompt_template = asc_842_prompt_template_structured(template_string)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        question_answerer = QuestionAnswererV2(
            vectorstore=PineconeVectorStore(
                index_name=index_name, embedding=embeddings
            ),
            namespace=namespace,
            prompt_template=prompt_template,
            retriever_question=retriever_question,
            use_llm_filter=use_llm_filter,
            lessor_question=lessor_question,
            date_question=date_question,
            boolean_question=boolean_question,
            temperature=temperature,
            use_listwise_rerank=use_listwise_rerank,
            use_embeddings_filter=use_embeddings_filter,
            embeddings=embeddings,
            chunks_filename=chunks_filename,
            lease_number=lease_number,
        )
        answer = question_answerer.process_answer()
        return answer
    except Exception as e:
        raise e
