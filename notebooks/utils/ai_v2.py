from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from openai_pinecone.services.question_answerer_v2 import QuestionAnswererV2
from openai_pinecone.utils.asc_842 import asc_842_prompt_template_structured


def get_answer_from_ai_v2(
    namespace: str,
    index_name: str,
    template_string: str | None = None,
    retriever_question: str | None = None,
):
    try:
        prompt_template = asc_842_prompt_template_structured(template_string)

        question_answerer = QuestionAnswererV2(
            vectorstore=PineconeVectorStore(
                index_name=index_name,
                embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
            ),
            namespace=namespace,
            prompt_template=prompt_template,
            retriever_question=retriever_question,
        )
        answer = question_answerer.process_answer()
        return answer
    except Exception as e:
        raise e
