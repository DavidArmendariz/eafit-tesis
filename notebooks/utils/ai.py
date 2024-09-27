from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from openai_pinecone.services.question_answerer import QuestionAnswerer
from openai_pinecone.utils.asc_842 import (
    asc_842_prompt_template,
    asc_842_prompt_template_structured,
)

from .constants import AvailableEmbeddingModels, AvailableModels


def get_answer_from_ai(
    questions: str,
    namespace: str,
    index_name: str,
    model: AvailableModels = "gpt-3.5-turbo",
    embedding_model: AvailableEmbeddingModels = "text-embedding-ada-002",
    supports_structured_output=False,
):
    try:
        prompt_template = (
            asc_842_prompt_template_structured
            if supports_structured_output
            else asc_842_prompt_template
        )

        if model == "gpt-3.5-turbo" and supports_structured_output:
            llm = ChatOpenAI(
                model=model,
                temperature=0,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
        else:
            llm = ChatOpenAI(model=model, temperature=0)

        question_answerer = QuestionAnswerer(
            vectorstore=PineconeVectorStore(
                index_name=index_name,
                embedding=OpenAIEmbeddings(model=embedding_model),
            ),
            namespace=namespace,
            llm=llm,
            prompt_template=prompt_template,
            supports_structured_output=supports_structured_output,
            model=model,
        )
        answer = question_answerer.process_answer(questions)
        return answer
    except Exception as e:
        raise e
