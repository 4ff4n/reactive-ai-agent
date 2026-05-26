"""
RAG Fallback Agent
──────────────────
Used when SQL generation fails or the question is ambiguous/non-SQL.
Indexes internal documents under data/docs/ with FAISS + OpenAI embeddings.
Retrieves top-k chunks and generates a context-aware answer.
"""
import logging
import os
from pathlib import Path
from typing import AsyncIterator

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

DOCS_DIR = Path(__file__).parent.parent.parent / "data" / "docs"
INDEX_DIR = Path(__file__).parent.parent.parent / "data" / "faiss_index"

RAG_SYSTEM_PROMPT = """You are a helpful e-commerce business analyst assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.
Be concise and professional.

Context:
{context}
"""


class RAGAgent:
    def __init__(self):
        self._vectorstore: FAISS | None = None
        self._embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
        )
        self._llm = ChatOpenAI(
            model=settings.model_fast,
            api_key=settings.openai_api_key,
            temperature=0.2,
        )

    async def _build_or_load_index(self) -> FAISS:
        """Load existing FAISS index or build from docs/."""
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        index_file = INDEX_DIR / "index.faiss"

        if index_file.exists():
            logger.info("Loading existing FAISS index from %s", INDEX_DIR)
            return FAISS.load_local(
                str(INDEX_DIR),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )

        logger.info("Building FAISS index from %s", DOCS_DIR)
        loader = DirectoryLoader(
            str(DOCS_DIR),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        logger.info("Indexing %d chunks", len(chunks))

        vs = FAISS.from_documents(chunks, self._embeddings)
        vs.save_local(str(INDEX_DIR))
        return vs

    async def ensure_ready(self) -> None:
        if self._vectorstore is None:
            self._vectorstore = await self._build_or_load_index()

    def _retrieve_context(self, question: str) -> str:
        assert self._vectorstore is not None
        docs = self._vectorstore.similarity_search(question, k=settings.top_k_rag)
        return "\n\n---\n\n".join(d.page_content for d in docs)

    async def answer(self, question: str, history: list, callbacks: list = None) -> str:
        await self.ensure_ready()
        context = self._retrieve_context(question)

        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT.format(context=context)),
            ("placeholder", "{history}"),
            ("human", "{question}"),
        ])
        chain = prompt | self._llm | StrOutputParser()
        invoke_config = {"callbacks": callbacks} if callbacks else {}
        return await chain.ainvoke({"question": question, "history": history}, config=invoke_config)

    async def answer_stream(self, question: str, history: list) -> AsyncIterator[str]:
        """Token-by-token streaming variant."""
        await self.ensure_ready()
        context = self._retrieve_context(question)

        streaming_llm = ChatOpenAI(
            model=settings.model_fast,
            api_key=settings.openai_api_key,
            temperature=0.2,
            streaming=True,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT.format(context=context)),
            ("placeholder", "{history}"),
            ("human", "{question}"),
        ])
        chain = prompt | streaming_llm | StrOutputParser()
        async for chunk in chain.astream({"question": question, "history": history}):
            yield chunk

    def invalidate_index(self) -> None:
        """Force rebuild on next request (call after adding new docs)."""
        self._vectorstore = None
        import shutil
        if INDEX_DIR.exists():
            shutil.rmtree(INDEX_DIR)
        logger.info("FAISS index invalidated — will rebuild on next request")


_rag_agent: RAGAgent | None = None


def get_rag_agent() -> RAGAgent:
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = RAGAgent()
    return _rag_agent
