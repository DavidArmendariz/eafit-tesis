from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_pinecone.vectorstores import PineconeVectorStore

from .types import QuestionResponse


class QuestionAnswerer:
    def __init__(
        self,
        vectorstore: PineconeVectorStore,
        namespace: str,
        llm: ChatOpenAI = ChatOpenAI(),
        prompt_template=PromptTemplate.from_template(""),
        supports_structured_output=False,
        model: Literal[
            "gpt-3.5-turbo", "gpt-4o-2024-08-06", "gpt-4o-mini"
        ] = "gpt-3.5-turbo",
    ):
        self.namespace = namespace
        self.vectorstore = vectorstore
        self.llm = llm
        self.prompt_template = prompt_template
        self.supports_structured_output = supports_structured_output
        self.model = model

    def process_answer(self, question):
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"namespace": self.namespace}
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        llm = self.llm
        retrieve_context = {
            "context": retriever | format_docs,
            "questions": RunnablePassthrough(),
        }

        if self.supports_structured_output and self.model != "gpt-3.5-turbo":
            llm = llm.with_structured_output(QuestionResponse, method="json_schema")
            rag_chain = retrieve_context | self.prompt_template | llm
        else:
            rag_chain = (
                retrieve_context | self.prompt_template | llm | StrOutputParser()
            )

        return rag_chain.invoke(question)
