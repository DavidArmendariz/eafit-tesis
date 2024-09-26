import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from openai_pinecone.services.question_answerer import QuestionAnswerer
from openai_pinecone.utils.asc_842 import (
    asc_842_prompt_template,
    asc_842_prompt_template_structured,
)

from .constants import AvailableEmbeddingModels, AvailableModels

load_dotenv()


def get_answer_from_ai(
    questions: str,
    namespace: str,
    model: AvailableModels = "gpt-3.5-turbo",
    embedding_model: AvailableEmbeddingModels = "text-embedding-ada-002",
    supports_structured_output=False,
    use_json_object=False,
    index_name=os.getenv("INDEX_NAME", ""),
):
    try:
        prompt_template = (
            asc_842_prompt_template_structured
            if supports_structured_output and not use_json_object
            else asc_842_prompt_template
        )
        question_answerer = QuestionAnswerer(
            vectorstore=PineconeVectorStore(
                index_name=index_name,
                embedding=OpenAIEmbeddings(model=embedding_model),
            ),
            namespace=namespace,
            llm=ChatOpenAI(model=model, temperature=0),
            prompt_template=prompt_template,
            supports_structured_output=supports_structured_output,
            use_json_object=use_json_object,
        )
        answer = question_answerer.process_answer(questions)
        return answer
    except Exception as e:
        raise e
