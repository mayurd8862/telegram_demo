from gtts import gTTS
import io

def text_to_speech(text: str):
    """Convert text to speech using gTTS and return raw audio bytes."""
    tts = gTTS(text=text, lang="en")
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.read()   # ✅ store raw bytes (not BytesIO)
