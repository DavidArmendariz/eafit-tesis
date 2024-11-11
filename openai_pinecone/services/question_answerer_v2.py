from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_filter import LLMChainFilter
from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter
from langchain.retrievers.document_compressors.listwise_rerank import LLMListwiseRerank
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
        lease_number: int,
        prompt_template=PromptTemplate.from_template(""),
        use_llm_filter=False,
        use_listwise_rerank=False,
        retriever_question: str | None = None,
        lessor_question=True,
        date_question=False,
        boolean_question=False,
        temperature=0.0,
        use_embeddings_filter=False,
        embeddings=OpenAIEmbeddings(model="text-embedding-3-large"),
        chunks_filename: str | None = None,
    ):
        self.namespace = namespace
        self.vectorstore = vectorstore
        self.prompt_template = prompt_template
        self.use_llm_filter = use_llm_filter
        self.retriever_question = retriever_question
        self.lessor_question = lessor_question
        self.date_question = date_question
        self.boolean_question = boolean_question
        self.temperature = temperature
        self.use_listwise_rerank = use_listwise_rerank
        self.use_embeddings_filter = use_embeddings_filter
        self.embeddings = embeddings
        self.chunks_filename = chunks_filename
        self.lease_number = lease_number

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

        if self.retriever_question:
            llm_for_contextual_compression = ChatOpenAI(
                model="gpt-4o-mini", temperature=self.temperature
            )
            if self.use_llm_filter:
                _filter = LLMChainFilter.from_llm(llm_for_contextual_compression)
            elif self.use_listwise_rerank:
                _filter = LLMListwiseRerank.from_llm(llm_for_contextual_compression)
            elif self.use_embeddings_filter:
                _filter = EmbeddingsFilter(
                    embeddings=self.embeddings, similarity_threshold=0.76
                )
            retriever = ContextualCompressionRetriever(
                base_compressor=_filter, base_retriever=retriever
            )
            chunks = retriever.invoke(self.retriever_question)
            if self.chunks_filename:
                with open(self.chunks_filename, "a") as f:
                    f.write(f"Chunks for lease {self.lease_number}\n")
                    f.write(f"Number of chunks retrieved: {len(chunks)}\n")
                    content = f"\n{'*' * 100}\n".join(
                        [
                            f"Document {i+1}:\n\n" + d.page_content
                            for i, d in enumerate(chunks)
                        ]
                    )
                    f.write(content)
                    f.write(f"\n\n{'^' * 100}\n\n")
            retrieve_context["context"] = retriever | format_docs
            rag_chain = retrieve_context | self.prompt_template | llm
            return rag_chain.invoke(self.retriever_question)

        rag_chain = retrieve_context | self.prompt_template | llm

        return rag_chain.invoke("")
