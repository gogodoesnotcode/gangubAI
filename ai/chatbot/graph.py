"""LangGraph agent graph definition – exact same logic as the original monolith."""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from ai.chatbot.config import MODEL_ID, SYSTEM_PROMPT
from ai.chatbot.tools import calculator, retrieve_context


# ── State schema ───────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class RAGResponse(BaseModel):
    explanation: str = Field(
        description="Detailed explanation answering the user's question based on the retrieved context"
    )
    citation: str = Field(
        description="Citation in the format: 'For more reference check <source>, <type> <number>' "
                    "e.g. 'For more reference check ECC_1_Introduction.pptx, Slide 3'"
    )


# ── LLM bindings ──────────────────────────────────────────
tools_list = [calculator, retrieve_context]
model = init_chat_model(MODEL_ID)
llm_with_tools = model.bind_tools(tools_list)
llm_for_rag = model.with_structured_output(RAGResponse)


# ── Chat node ─────────────────────────────────────────────
def chatNode(state: ChatState) -> ChatState:
    messages = state["messages"]
    all_messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    # Check if retrieve_context was called in the most recent tool round-trip
    rag_was_called = False
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            break
        if isinstance(msg, ToolMessage) and msg.name == "retrieve_context":
            rag_was_called = True
            break

    if rag_was_called:
        response: RAGResponse = llm_for_rag.invoke(all_messages)
        formatted = f"{response.explanation}\n\n**For more reference:** {response.citation}"
        return {"messages": [AIMessage(content=formatted)]}

    response = llm_with_tools.invoke(all_messages)
    return {"messages": [response]}


# ── Build graph ───────────────────────────────────────────
tool_node = ToolNode(tools_list)

graph = StateGraph(ChatState)
graph.add_node("chat", chatNode)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")

# For LangGraph API – compile without checkpointer (API handles persistence)
agent = graph.compile()
