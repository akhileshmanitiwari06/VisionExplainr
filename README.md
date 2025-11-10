
# VisionExplainr

VisionExplainr is a starter project that explains what is happening in short videos using
MediaPipe (pose + hands) and simple heuristics. It produces a timeline of events and human-friendly
explanations, and can optionally use OpenAI to polish text and gTTS for audio narration.

## Quick start
1. Create & activate venv:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Add a short test video at `example_inputs/sample_video.mp4` (<= 30s recommended).
4. Run app:
   ```bash
   streamlit run app.py
   ```
5. (Optional) To enable OpenAI polishing, set `OPENAI_API_KEY` in your environment.

## Notes
- gTTS requires internet to synthesize audio.
- MediaPipe works better with clear, well-lit videos.
