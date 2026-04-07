from faster_whisper import WhisperModel
import os
import streamlit as st
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

@st.cache_resource
def load_model():
    return WhisperModel("tiny", device="cpu", compute_type="int8")

model = load_model()

def transcribe_audio(audio_path:str):
    """
   Transcribes audio file using whisper. Returns timestamps transcript(str) and duration(sec)
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError("Audio file not found.")
    segments, info = model.transcribe(audio_path)

    transcript_lines = []
    duration = 0

    for segment in segments:
        start_time = int(segment.start)
        minutes = start_time // 60
        seconds = start_time % 60
        timestamp = f"[{minutes:02d}:{seconds:02d}]"
        text = segment.text.strip()

        transcript_lines.append(f"{timestamp} {text}")
    
        duration = round(segment.end)
    transcript = "\n".join(transcript_lines)
    print(len(transcript))
    return transcript, duration