# VisionExplainr
AI that explains what’s happening inside a video in plain English — step-by-step.

🎯 Goal

Computer Vision + LLM = Scene-by-scene description generator.

💡 Use Case

Video analytics, sports, CCTV interpretation, or accessibility for the visually impaired.

🧩 Architecture

Frame Extraction (1 frame/sec)

Action Recognition (e.g., YOLO + DeepSORT)

Event Detection (goal, jump, run, fight, etc.)

LLM Commentary Generation

Audio Narration (TTS).

⚙️ Tech Stack

OpenCV, YOLOv8, MediaPipe (action detection)

OpenAI GPT / Llama-3 (caption + commentary)

gTTS / Coqui (voice output)

Flask / Streamlit

🚀 Steps

Extract video frames.

Detect human poses/actions.

Convert detections into structured event list.

Generate description: “Person A runs towards B and throws ball.”

Convert to audio + overlay on video.
