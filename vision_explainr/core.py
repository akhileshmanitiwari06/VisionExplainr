
# vision_explainr/core.py
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import math
import os
from gtts import gTTS
import tempfile
import json
from dotenv import load_dotenv
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY", None)
try:
    if OPENAI_KEY:
        import openai
        openai.api_key = OPENAI_KEY
except Exception:
    openai = None

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

def extract_frames(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video: " + video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(video_fps / fps)))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append((int(cap.get(cv2.CAP_PROP_POS_MSEC)), frame.copy()))
        idx += 1
    cap.release()
    return frames

def detect_pose_and_hands(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    res = {"pose": None, "hands": []}
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        r = pose.process(img_rgb)
        if r.pose_landmarks:
            pts = []
            for lm in r.pose_landmarks.landmark:
                pts.append((lm.x, lm.y, lm.z))
            res["pose"] = pts
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        rh = hands.process(img_rgb)
        if rh.multi_hand_landmarks:
            for hand_landmarks in rh.multi_hand_landmarks:
                pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                res["hands"].append(pts)
    return res

def estimate_speed(last_positions, window=3, fps=1.0):
    if len(last_positions) < 2:
        return 0.0
    speeds = []
    for i in range(1, len(last_positions)):
        t0, p0 = last_positions[i-1]
        t1, p1 = last_positions[i]
        dt = max((t1 - t0) / 1000.0, 1e-3)
        dx = (p1[0] - p0[0])
        dy = (p1[1] - p0[1])
        dist = math.hypot(dx, dy)
        speeds.append(dist / dt)
    if not speeds:
        return 0.0
    return sum(speeds) / len(speeds)

def bbox_center_of_pose(pose_landmarks):
    if not pose_landmarks:
        return None
    xs = [p[0] for p in pose_landmarks if p[0] is not None]
    ys = [p[1] for p in pose_landmarks if p[1] is not None]
    if not xs or not ys:
        return None
    return (sum(xs)/len(xs), sum(ys)/len(ys))

def detect_actions_over_frames(frames, fps=1.0):
    events = []
    person_centers = deque(maxlen=8)
    last_event = None
    center_history = []
    for (t_ms, frame) in frames:
        det = detect_pose_and_hands(frame)
        center = bbox_center_of_pose(det.get("pose"))
        h,w = frame.shape[:2]
        center_px = None
        if center:
            center_px = (center[0]*w, center[1]*h)
        if center_px:
            center_history.append((t_ms, center_px))
            if len(center_history) > 6:
                center_history.pop(0)
        speed = estimate_speed(center_history, window=6, fps=fps)
        hands = det.get("hands", [])
        hands_up = False
        if det.get("pose"):
            try:
                shoulder_y = (det['pose'][11][1] + det['pose'][12][1]) / 2.0
                wrist_positions = []
                if len(det['pose']) > 16:
                    wrist_positions = [det['pose'][15], det['pose'][16]]
                for wp in wrist_positions:
                    if wp and wp[1] < shoulder_y - 0.02:
                        hands_up = True
            except Exception:
                hands_up = False
        fall_detected = False
        if len(center_history) >= 3:
            ys = [p[1] for (_,p) in center_history]
            if max(ys) - min(ys) > h * 0.15:
                fall_detected = True
        label = None
        confidence = 0.5
        if center is None:
            label = "No person detected"
        else:
            if fall_detected:
                label = "Possible fall / sudden drop"
                confidence = 0.8
            elif hands_up:
                label = "Hands raised / waving / cheering"
                confidence = 0.8
            else:
                if speed < 20:
                    label = "Standing / stationary"
                    confidence = 0.6
                elif speed < 100:
                    label = "Walking / light movement"
                    confidence = 0.7
                else:
                    label = "Running / fast movement"
                    confidence = 0.8
        if last_event is None:
            last_event = {"start": t_ms, "end": t_ms, "label": label, "confidence": confidence}
        else:
            if label == last_event["label"]:
                last_event["end"] = t_ms
            else:
                events.append(last_event)
                last_event = {"start": t_ms, "end": t_ms, "label": label, "confidence": confidence}
    if last_event:
        events.append(last_event)
    for e in events:
        dur_s = max(1, int((e['end'] - e['start'])/1000))
        e['summary'] = f"{e['label']} for about {dur_s} sec (confidence {e['confidence']:.2f})"
    return events

def generate_explanations(events, use_llm=False):
    texts = [e['summary'] for e in events]
    if use_llm and openai is not None:
        prompt = "You are a concise assistant. Convert the following event summaries into polished single-sentence explanations suitable for a video narrator. Keep them short.\n\n"
        for i,t in enumerate(texts):
            prompt += f"{i+1}. {t}\n"
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content":prompt}],
                max_tokens=500,
                temperature=0.2
            )
            out = resp['choices'][0]['message']['content'].strip()
            lines = [ln.strip("- ").strip() for ln in out.splitlines() if ln.strip()]
            if len(lines) >= len(texts):
                return lines[:len(texts)]
            else:
                return [f"Event {i+1}: {t}" for i,t in enumerate(texts)]
        except Exception as ex:
            print("LLM failed:", ex)
            return [f"Event {i+1}: {t}" for i,t in enumerate(texts)]
    else:
        polished = []
        for i,t in enumerate(texts):
            s = t.replace("Possible fall / sudden drop", "A sudden fall or drop was detected")
            s = s.replace("Hands raised / waving / cheering", "Someone raised hands — looks like a wave or cheer")
            s = s.replace("Standing / stationary", "Person standing or stationary")
            s = s.replace("Walking / light movement", "Person walking")
            s = s.replace("Running / fast movement", "Person running or moving quickly")
            polished.append(s)
        return polished

def text_to_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmpname = tmp.name
        tmp.close()
        tts.save(tmpname)
        with open(tmpname, 'rb') as f:
            data = f.read()
        try:
            os.unlink(tmpname)
        except Exception:
            pass
        return data
    except Exception as e:
        print("TTS failed:", e)
        return None

def explain_video(video_path, fps=1, use_llm=False, tts_lang='en'):
    frames = extract_frames(video_path, fps=fps)
    if not frames:
        return {"frames_count":0, "events":[], "explanations":[], "tts": None}
    events = detect_actions_over_frames(frames, fps=fps)
    explanations = generate_explanations(events, use_llm=use_llm)
    tts_bytes = None
    full_narration = " . ".join(explanations)
    if tts_lang and full_narration.strip():
        tts_bytes = text_to_speech(full_narration, lang=tts_lang)
    return {"frames_count": len(frames), "events": events, "explanations": explanations, "tts": tts_bytes}
