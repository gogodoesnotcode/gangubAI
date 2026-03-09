"""Centralised configuration for GangubAI chatbot."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Model ──────────────────────────────────────────────────
MODEL_ID = "google_genai:gemini-2.5-flash-lite"

# ── Embeddings ─────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# ── Vector store ───────────────────────────────────────────
CHROMA_COLLECTION = "gangubai_collection"
PERSIST_DIRECTORY = ".gangubai_db_hf/"

# ── RAG document sources (relative to repo root) ──────────
RAG_FILES = [
    os.path.join("ai", "chatbot", "Unit1", "ECC_1_Introduction to GenAI and Its Application.pptx"),
    os.path.join("ai", "chatbot", "Unit1", "ECC_2_LLM basics and Evolution.pptx"),
    os.path.join("ai", "chatbot", "Unit1", "ECC_3_NLP Basics_Terminologies_TaskOverview.pptx"),
]

# ── System prompt ──────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are GangubAI, a helpful AI assistant with access to a knowledge base. "
    "Use the retrieve_context tool to search for relevant information before answering questions. "
    "Always cite your sources using the metadata provided (source, type, and number). "
    "If you can't find relevant information, say so honestly."
)
