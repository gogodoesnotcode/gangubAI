"""RAG retrieval tool – talks to the Chroma vectorstore."""

from langchain_core.tools import tool

from ai.chatbot.document_processor import get_vectorstore

# Module-level vectorstore singleton (loaded once on first import)
vectorstore = get_vectorstore()


@tool
def retrieve_context(query: str) -> str:
    """Retrieve information from the Gangubai knowledge base to help answer a query.

    Use this tool when you need to answer questions about topics in the knowledge base.

    Args:
        query: The question or topic to search for in the knowledge base

    Returns:
        Relevant information from the knowledge base with source citations
    """
    retrieved_docs = vectorstore.similarity_search(query, k=3)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized
