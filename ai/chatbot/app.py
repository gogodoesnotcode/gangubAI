"""Interactive CLI entrypoint – GangubAI Chatbot with RAG."""

import json
import subprocess
import time

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver, sqlite3

from ai.frontend.bridge.local_queue import EmotionPublisher
from ai.chatbot.graph import graph


_EMOTION_PUBLISHER = EmotionPublisher()
_ROS_EMOTION_TOPIC = "/robot_emotion"


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
            _emit_emotion(str(emotion), source="chatbot")
            print("=" * 60)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! GangubAI shutting down...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different question.")


def _emit_emotion(emotion: str, source: str = "chatbot") -> None:
    normalized = emotion.strip().lower() or "neutral"
    _EMOTION_PUBLISHER.publish(normalized, source=source)
    _publish_emotion_to_ros(normalized, source=source)


def _publish_emotion_to_ros(emotion: str, source: str) -> None:
    message_json = json.dumps(
        {
            "emotion": emotion,
            "source": source,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "payload": "",
        },
        ensure_ascii=True,
    )
    ros_payload = "{data: '" + message_json + "'}"

    try:
        subprocess.Popen(
            [
                "ros2",
                "topic",
                "pub",
                "--once",
                _ROS_EMOTION_TOPIC,
                "std_msgs/msg/String",
                ros_payload,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return


if __name__ == "__main__":
    main()
