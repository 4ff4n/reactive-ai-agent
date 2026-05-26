"""
Speech-to-Text via OpenAI Whisper API.
Accepts audio bytes (WebM/MP3/WAV) and returns transcribed text.
Uses pydub + ffmpeg to normalise to 16kHz mono WAV before sending.

considerations:
  - Forces language="en" to prevent Whisper hallucinating in wrong languages
  - Rejects audio shorter than MIN_DURATION_SECONDS (likely silence)
  - Filters hallucinated one-word outputs like "you", "thank you", "thanks"
"""
import io
import logging
import tempfile
from pathlib import Path

from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

MIN_DURATION_SECONDS = 1.0   # reject clips shorter than this
MIN_WORD_COUNT       = 2     # reject transcripts with fewer words

# Whisper hallucinates these on silence — treat as failed transcription
HALLUCINATION_PHRASES = {
    "you", "thank you", "thanks", "thank you.", "thanks.",
    "you.", "bye", "bye.", "ok", "okay", "ok.", "okay.",
    ".", "..", "...", "um", "uh", "hmm",
}


def _convert_to_wav(audio_bytes: bytes, src_format: str = "webm") -> tuple[bytes, float]:
    """
    Convert audio to 16kHz mono WAV using pydub + ffmpeg.
    Returns (wav_bytes, duration_seconds).
    """
    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=src_format)
        duration = len(audio) / 1000.0   # milliseconds → seconds

        audio = audio.set_frame_rate(16_000).set_channels(1)
        buf = io.BytesIO()
        audio.export(buf, format="wav")
        return buf.getvalue(), duration
    except Exception as e:
        logger.warning("pydub conversion failed (%s); using raw bytes", e)
        return audio_bytes, 999.0   # assume long enough if conversion fails


async def transcribe(audio_bytes: bytes, audio_format: str = "webm") -> dict:
    """
    Transcribe audio bytes using OpenAI Whisper API.
    Returns {"text": str, "confidence": str, "language": str}.
    """
    import openai

    # Convert and check duration
    wav_bytes, duration = _convert_to_wav(audio_bytes, src_format=audio_format)
    logger.info("Audio duration: %.2fs", duration)

    if duration < MIN_DURATION_SECONDS:
        logger.warning("Audio too short (%.2fs < %.1fs) — skipping transcription", duration, MIN_DURATION_SECONDS)
        return {
            "text": "",
            "language": "unknown",
            "confidence": "rejected",
            "error": f"Audio too short ({duration:.1f}s) — please hold the mic button and speak clearly",
        }

    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = Path(tmp.name)

    try:
        with open(tmp_path, "rb") as f:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                language="en",          # force English — prevents hallucinations in wrong languages
                temperature=0,          # deterministic output
            )

        text = response.text.strip()
        language = getattr(response, "language", "en")

        # Filter hallucinations
        if text.lower().strip(".,!? ") in HALLUCINATION_PHRASES:
            logger.warning("Whisper hallucination detected: '%s' — treating as empty", text)
            return {
                "text": "",
                "language": language,
                "confidence": "hallucination",
                "error": "Could not understand audio — please speak clearly and try again",
            }

        word_count = len(text.split())
        if word_count < MIN_WORD_COUNT:
            logger.warning("Transcript too short (%d words): '%s'", word_count, text)
            return {
                "text": "",
                "language": language,
                "confidence": "too_short",
                "error": f"Transcript too short ('{text}') — please speak a full sentence",
            }

        logger.info("Transcribed (%s, %.1fs): %s", language, duration, text[:80])
        return {"text": text, "language": language, "confidence": "high"}

    except Exception as e:
        logger.error("Whisper transcription failed: %s", e)
        return {"text": "", "language": "unknown", "confidence": "failed", "error": str(e)}
    finally:
        tmp_path.unlink(missing_ok=True)
