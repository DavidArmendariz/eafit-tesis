from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_filter import LLMChainFilter
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone.vectorstores import PineconeVectorStore

from .types import (
    QuestionResponseAll,
    QuestionResponseBoolean,
    QuestionResponseDate,
    QuestionResponseStr,
)


class QuestionAnswererV2:
    def __init__(
        self,
        vectorstore: PineconeVectorStore,
        namespace: str,
        prompt_template=PromptTemplate.from_template(""),
        use_llm_filter=False,
        retriever_question: str | None = None,
        lessor_question=True,
        date_question=False,
        boolean_question=False,
    ):
        self.namespace = namespace
        self.vectorstore = vectorstore
        self.prompt_template = prompt_template
        self.use_llm_filter = use_llm_filter
        self.retriever_question = retriever_question
        self.lessor_question = lessor_question
        self.date_question = date_question
        self.boolean_question = boolean_question

    def process_answer(self):
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"namespace": self.namespace}
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        retrieve_context = {
            "context": retriever | format_docs,
        }

        structure = QuestionResponseAll
        if self.lessor_question:
            structure = QuestionResponseStr
        elif self.date_question:
            structure = QuestionResponseDate
        elif self.boolean_question:
            structure = QuestionResponseBoolean

        llm = llm.with_structured_output(structure, method="json_schema")

        if self.use_llm_filter and self.retriever_question:
            _filter = LLMChainFilter.from_llm(
                ChatOpenAI(model="gpt-4o-mini", temperature=0)
            )
            retriever = ContextualCompressionRetriever(
                base_compressor=_filter, base_retriever=retriever
            )
            retrieve_context["context"] = retriever | format_docs
            rag_chain = retrieve_context | self.prompt_template | llm
            return rag_chain.invoke(self.retriever_question)

        rag_chain = retrieve_context | self.prompt_template | llm

        return rag_chain.invoke("")
