"""LangGraph agent graph definition – exact same logic as the original monolith."""

from enum import Enum
import re
from typing import Annotated, Literal, Optional, TypedDict
from uuid import uuid4

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from ai.chatbot.config import MODEL_ID, SYSTEM_PROMPT
from ai.chatbot.tools import calculator, retrieve_context, move_robot, set_wander_mode


# ── Emotion enum ───────────────────────────────────────────
class Emotion(str, Enum):
    """Robot emotions – each maps to a face animation / physical action."""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    CURIOUS = "curious"
    EXCITED = "excited"
    CONFUSED = "confused"
    NEUTRAL = "neutral"
    THINKING = "thinking"


EMOTION_VALUES = Literal[
    "happy", "sad", "angry", "curious",
    "excited", "confused", "neutral", "thinking",
]


# ── State schema ───────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_emotion: str  # one of Emotion values


# ── Structured output schemas ──────────────────────────────
class ChatResponse(BaseModel):
    """Standard chatbot reply with an associated emotion."""
    content: str = Field(
        description="The chatbot's reply to the user."
    )
    emotion: EMOTION_VALUES = Field(
        description="The single emotion that best matches this response. "
                    "One of: happy, sad, angry, curious, excited, confused, neutral, thinking.",
        default="neutral",
    )


class RAGResponse(BaseModel):
    """RAG-grounded reply with citation and emotion."""
    explanation: str = Field(
        description="Detailed explanation answering the user's question based on the retrieved context"
    )
    citation: str = Field(
        description="Citation in the format: 'For more reference check <source>, <type> <number>' "
                    "e.g. 'For more reference check ECC_1_Introduction.pptx, Slide 3'"
    )
    emotion: EMOTION_VALUES = Field(
        description="The single emotion that best matches this response. "
                    "One of: happy, sad, angry, curious, excited, confused, neutral, thinking.",
        default="neutral",
    )


# ── LLM bindings ──────────────────────────────────────────
tools_list = [calculator, retrieve_context, move_robot, set_wander_mode]
# model = init_chat_model(MODEL_ID)
model = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)
llm_with_tools = model.bind_tools(tools_list)
llm_for_rag = model.with_structured_output(RAGResponse)
llm_for_chat = model.with_structured_output(ChatResponse)


def _latest_user_query(messages: list[BaseMessage]) -> str:
    """Return the latest HumanMessage content as a plain string."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content).strip()
    return ""


def _looks_like_calculator_query(user_query: str) -> bool:
    q = user_query.lower().strip()
    if not q:
        return False

    if any(keyword in q for keyword in ("calculate", "compute", "plus", "minus", "multiply", "divide")):
        return True

    return bool(re.fullmatch(r"[0-9\s\+\-\*\/\(\)\.=:%]+", q))


def _looks_like_robot_action_query(user_query: str) -> bool:
    q = user_query.lower().strip()
    robot_keywords = (
        "move",
        "turn",
        "rotate",
        "forward",
        "backward",
        "left",
        "right",
        "stop",
        "navigate",
        "motor",
        "gripper",
        "robot",
    )
    return any(keyword in q for keyword in robot_keywords)


def _looks_like_smalltalk_query(user_query: str) -> bool:
    q = user_query.lower().strip()
    smalltalk_cues = (
        "joke",
        "funny",
        "humor",
        "how are you",
        "who are you",
        "your name",
        "hello",
        "hi",
        "thanks",
        "thank you",
    )
    return any(cue in q for cue in smalltalk_cues)


def _should_force_retrieve(user_query: str) -> bool:
    """Fallback rule: enforce retrieve_context for knowledge-style questions."""
    q = user_query.lower().strip()
    if not q:
        return False

    if (
        _looks_like_calculator_query(q)
        or _looks_like_robot_action_query(q)
        or _looks_like_smalltalk_query(q)
    ):
        return False

    # Keep this intentionally strict. We only force retrieval when the
    # user is clearly asking for instructional/knowledge-grounded content.
    retrieval_cues = (
        "slides",
        "notes",
        "ppt",
        "pptx",
        "chapter",
        "unit",
        "from the material",
        "from the slides",
        "according to the material",
        "according to the slides",
    )

    has_retrieval_cue = any(cue in q for cue in retrieval_cues)
    if not has_retrieval_cue:
        return False

    return len(q.split()) >= 4


# ── Chat node ─────────────────────────────────────────────
def chatNode(state: ChatState) -> ChatState:
    messages = state["messages"]
    all_messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    latest_user_query = _latest_user_query(messages)

    # Small-talk should stay conversational and not hit retrieval tools.
    if _looks_like_smalltalk_query(latest_user_query):
        response: ChatResponse = llm_for_chat.invoke(all_messages)
        return {
            "messages": [AIMessage(content=response.content)],
            "current_emotion": response.emotion,
        }

    # Check if retrieve_context was called in the most recent tool round-trip
    rag_was_called = False
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            break
        if isinstance(msg, ToolMessage) and msg.name == "retrieve_context":
            rag_was_called = True
            break

    if rag_was_called:
        # Extract the retrieved context from the ToolMessage and build a
        # clean prompt so the structured-output model doesn't see raw
        # tool-call artefacts (which confuse Gemini into generating code).
        retrieved_context = ""
        user_question = ""
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and msg.name == "retrieve_context":
                retrieved_context = msg.content
            if isinstance(msg, (HumanMessage,)):
                user_question = msg.content
                break

        rag_messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"User question: {user_question}\n\n"
                    f"Retrieved context:\n{retrieved_context}\n\n"
                    "Using the retrieved context, answer the user's question. "
                    "Explain in your own words — simple and clear."
                )
            ),
        ]
        response: RAGResponse = llm_for_rag.invoke(rag_messages)
        formatted = f"{response.explanation}\n\n {response.citation}"
        return {
            "messages": [AIMessage(content=formatted)],
            "current_emotion": response.emotion,
        }

    # If the LLM decides to call a tool, fall through to the tool-calling path
    tool_response = llm_with_tools.invoke(all_messages)
    if tool_response.tool_calls:
        return {"messages": [tool_response]}

    # Gemini occasionally returns "I'll look it up" style text without
    # actually emitting a tool call. Force a real retrieve_context call so
    # RAG still executes end-to-end.
    if _should_force_retrieve(latest_user_query):
        forced_tool_call = {
            "name": "retrieve_context",
            "args": {"query": latest_user_query},
            "id": f"force_retrieve_{uuid4().hex}",
            "type": "tool_call",
        }
        return {
            "messages": [AIMessage(content="", tool_calls=[forced_tool_call])],
            "current_emotion": Emotion.THINKING.value,
        }

    # No tool calls → get a structured response with emotion
    response: ChatResponse = llm_for_chat.invoke(all_messages)
    return {
        "messages": [AIMessage(content=response.content)],
        "current_emotion": response.emotion,
    }


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
