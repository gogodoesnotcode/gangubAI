# GangubAI: Voice + RAG + ROS Robot Control

GangubAI is a multimodal assistant that combines:
- Retrieval-augmented question answering over technical learning materials (PDF/PPTX)
- A full voice loop (wake word, recording, transcription, response generation, TTS)
- ROS 2 robot control tools for direct movement and autonomous wander mode

## 1) Project Scope and Relevance

This project focuses on intelligent document comparison with structured extraction, normalization, semantic similarity, OCR, and LLM-assisted change analysis.

### What is already implemented in this repository
- Structured document ingestion foundation:
  - PDF parsing with `PyMuPDF` in `ai/chatbot/document_processor.py`
  - PPTX slide parsing with `python-pptx` in `ai/chatbot/document_processor.py`
- Segmentation and normalization baseline:
  - Recursive chunking with overlap via `RecursiveCharacterTextSplitter`
  - Metadata-enriched chunks (`source`, `type`, `number`) embedded into retrieval content
- Semantic retrieval foundation:
  - Embeddings (`BAAI/bge-large-en-v1.5`) + Chroma vector DB in `ai/chatbot/document_processor.py`
  - Similarity search tool (`retrieve_context`) in `ai/chatbot/tools/rag_tools.py`
- LLM-assisted reasoning layer:
  - LangGraph orchestration and structured outputs in `ai/chatbot/graph.py`
  - Tool-based decision routing between small talk, RAG, calculator, and ROS actions

## 2) Current End-to-End Functionality (General)

GangubAI currently supports the following runtime flow:
1. Wake word detection (`ai/voice/wake_word.py`) or push-to-talk fallback
2. Audio recording with silence-based stopping (`ai/voice/recorder.py`)
3. Speech-to-text using `whisper.cpp` (`ai/voice/transcriber.py`)
4. Response generation using LangGraph + tools (`ai/chatbot/graph.py`)
5. Emotion extraction from state and optional emission hooks (`ai/voice/app.py`)
6. Speech output with Piper TTS worker queue (`ai/voice/tts.py`)
7. Optional ROS action execution via tools (`ai/chatbot/tools/ros_tools.py`)

## 3) Architecture Overview

### AI / Document pipeline
- `ai/chatbot/document_processor.py`
  - Parses PDF/PPTX
  - Splits content into semantic chunks
  - Builds or loads Chroma vector store
- `ai/chatbot/tools/rag_tools.py`
  - Performs top-k similarity retrieval
- `ai/chatbot/graph.py`
  - LangGraph state machine
  - Structured response schemas (`ChatResponse`, `RAGResponse`)
  - Tool routing and fallback retrieval policy

### Voice pipeline
- `ai/voice/app.py`: full orchestration loop
- `ai/voice/wake_word.py`: OpenWakeWord integration + keyboard fallback
- `ai/voice/recorder.py`: adaptive/ptt recording
- `ai/voice/transcriber.py`: whisper.cpp wrapper
- `ai/voice/tts.py`: Piper streaming TTS with sentence chunking
- `ai/voice/config.py`: centralized runtime config

### ROS pipeline
- `ros2_ws/src/gangubai_control/gangubai_control/motor_controller_node.py`
  - Hardware mode (RPi GPIO + L298N)
  - Simulation mode (publishes `cmd_vel` for Gazebo)
- `ros2_ws/src/gangubai_control/gangubai_control/wander_controller_node.py`
  - Autonomous wander state machine
  - Cliff safety checks + edge recovery
  - Optional optical-flow stuck heuristic
- `ros2_ws/src/gangubai_control/launch/motor_control.launch.py`
  - Launches simulation/hardware stack + wander controller

## 4) Implemented Technical Mapping

### A) Extract structured content from PDFs
- Text extraction from PDFs and per-page metadata tagging

### B) Segment and normalize content
- Chunking with overlap
- Metadata-aware chunk formatting

### C) Semantic + OCR-based comparison
- Embedding-based semantic retrieval groundwork

### D) LLM-assisted change classification
- LLM orchestration with structured outputs and tool control

## 5) Setup and Run

## Prerequisites
- Python 3.10+
- ROS 2 Humble (for robot/simulation modules)
- Optional but used in voice mode:
  - `whisper.cpp` binary and model
  - Piper binary and voice model
  - OpenWakeWord model assets

### Python dependencies
There is no single locked dependency file at repo root yet.
Install the core libraries used by this project:

```bash
pip install \
  python-dotenv langchain langgraph langchain-core langchain-groq \
  chromadb langchain-community langchain-huggingface \
  sentence-transformers pymupdf python-pptx \
  sounddevice numpy scipy openwakeword
```

### Run chatbot (text mode)
```bash
cd gangubai_ws/gangubAI
python3 -m ai.chatbot.app
```

### Run the Pygame face frontend
```bash
cd gangubai_ws/gangubAI
pip install -r requirements.txt
python3 -m ai.frontend.app
```

The voice and chatbot entrypoints now broadcast emotion updates locally, so the
frontend will react to live turns when they are running at the same time.

To subscribe through ROS instead of local UDP:

```bash
python3 -m ai.frontend.app --ros-bridge --ros-emotion-topic /robot_emotion
```

### Run full voice loop
```bash
cd gangubai_ws/gangubAI
python3 -m ai.voice.app
```

Useful options:
```bash
python3 -m ai.voice.app --ptt
```

### Build and launch ROS stack
```bash
cd gangubai_ws/gangubAI/ros2_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
ros2 launch gangubai_control motor_control.launch.py simulate:=true wander_require_cliff_data:=false
```

### Run frontend + backend AI + Gazebo together
Use 3 terminals (4 if you also want text chatbot):

Terminal 1 (ROS + Gazebo):

```bash
cd ~/gangubai_ws/gangubAI/ros2_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
ros2 launch gangubai_control motor_control.launch.py simulate:=true wander_require_cliff_data:=false
```

Terminal 2 (Frontend face UI):

```bash
cd ~/gangubai_ws/gangubAI
source .venv/bin/activate
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
python3 -m ai.frontend.app --ros-bridge --ros-emotion-topic /robot_emotion
```

Terminal 3 (Backend AI voice loop):

```bash
cd ~/gangubai_ws/gangubAI
source .venv/bin/activate
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
python3 -m ai.voice.app --ptt
```

Optional Terminal 4 (text chatbot backend instead of voice):

```bash
cd ~/gangubai_ws/gangubAI
source .venv/bin/activate
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
python3 -m ai.chatbot.app
```

### Run frontend and backend together (single terminal)
If you want to stay on the Pygame frontend screen all the time, use:

```bash
cd ~/gangubai_ws/gangubAI
source .venv/bin/activate
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
python3 -m ai.launcher.run_face_backend --ros-bridge --backend-mode wakeword --fullscreen
```

Notes:
- This starts `ai.voice.app` in the background with `--no-keyboard-ptt` so wakeword works without terminal focus.
- Backend logs are written to `voice_backend.log` in the current directory.

## 6) Relevant Project Paths

- `ai/chatbot/document_processor.py`
- `ai/chatbot/graph.py`
- `ai/chatbot/tools/rag_tools.py`
- `ai/chatbot/tools/ros_tools.py`
- `ai/voice/app.py`
- `ai/voice/transcriber.py`
- `ai/voice/tts.py`
- `ai/voice/wake_word.py`
- `ros2_ws/src/gangubai_control/gangubai_control/motor_controller_node.py`
- `ros2_ws/src/gangubai_control/gangubai_control/wander_controller_node.py`

## 7) Notes

- The repository intentionally ignores heavy local assets (`piper/`, `whisper.cpp/`, `wakeword_models/`) via `.gitignore`.
- Configure paths for local binaries/models in `ai/voice/config.py`.
- For safety, keep `wander_require_cliff_data:=true` on real hardware deployments.
