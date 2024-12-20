from typing import Literal

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_filter import LLMChainFilter
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
        use_llm_filter=False,
        retriever_question: str | None = None,
    ):
        self.namespace = namespace
        self.vectorstore = vectorstore
        self.llm = llm
        self.prompt_template = prompt_template
        self.supports_structured_output = supports_structured_output
        self.model = model
        self.use_llm_filter = use_llm_filter
        self.retriever_question = retriever_question

    def process_answer(self, questions):
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

        if self.use_llm_filter and self.retriever_question:
            _filter = LLMChainFilter.from_llm(llm)
            retriever = ContextualCompressionRetriever(
                base_compressor=_filter, base_retriever=retriever
            )
            compressed_documents = retriever.invoke(self.retriever_question)
            retrieve_context["context"] = format_docs(compressed_documents)

        if self.supports_structured_output and self.model != "gpt-3.5-turbo":
            llm = llm.with_structured_output(QuestionResponse, method="json_schema")
            rag_chain = retrieve_context | self.prompt_template | llm
        else:
            rag_chain = (
                retrieve_context | self.prompt_template | llm | StrOutputParser()
            )

        return rag_chain.invoke(questions)
