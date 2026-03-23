"""Interactive CLI entrypoint – GangubAI Chatbot with RAG."""

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver, sqlite3

from ai.chatbot.graph import graph


def main():
    conn = sqlite3.connect(database="gangubai_chatbot_history.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)
    chatbot = graph.compile(checkpointer=checkpointer)

    print("\n" + "=" * 60)
    print("💬 GangubAI Chatbot - Type 'quit' or 'exit' to stop")
    print("=" * 60)

    config = {"configurable": {"thread_id": "2"}}

    while True:
        try:
            user_query = input("\n❓ Your question: ").strip()

            if user_query.lower() in ["quit", "exit", "q", "bye"]:
                print("\n👋 Goodbye! GangubAI shutting down...")
                break

            if not user_query:
                continue

            print(f"\n{'=' * 60}")
            print(f"❓ Question: {user_query}")
            print("=" * 60)
            print("\n💭 GangubAI is thinking...\n")

            # Stream the response
            final_state = None
            for event in chatbot.stream(
                {"messages": [HumanMessage(content=user_query)]},
                config=config,
                stream_mode="values",
            ):
                event["messages"][-1].pretty_print()
                final_state = event

            emotion = (final_state or {}).get("current_emotion", "neutral")
            print(f"\n🎭 Emotion: {emotion}")
            print("=" * 60)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! GangubAI shutting down...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different question.")


if __name__ == "__main__":
    main()
