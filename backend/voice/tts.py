"""
Text-to-Speech module.
Primary:  gTTS  (Google TTS cloud API, good quality, easy)
Fallback: pyttsx3 (offline, robotic but zero deps)
Optional: Coqui TTS (high quality, offline, more setup required)

Returns MP3 bytes that can be streamed over WebSocket or HTTP.
"""
import io
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class TTSEngine(str, Enum):
    GTTS   = "gtts"
    PYTTSX = "pyttsx3"
    COQUI  = "coqui"


async def synthesise(
    text: str,
    engine: TTSEngine = TTSEngine.GTTS,
    language: str = "en",
) -> bytes:
    """
    Convert text to speech. Returns MP3 bytes.
    """
    if engine == TTSEngine.GTTS:
        return await _gtts(text, language)
    elif engine == TTSEngine.COQUI:
        return await _coqui(text)
    else:
        return await _pyttsx(text)


async def _gtts(text: str, language: str = "en") -> bytes:
    try:
        from gtts import gTTS
        import asyncio

        def _synth():
            tts = gTTS(text=text, lang=language, slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            return buf.getvalue()

        # Run in executor to avoid blocking the event loop
        loop = __import__("asyncio").get_event_loop()
        audio_bytes = await loop.run_in_executor(None, _synth)
        logger.info("gTTS synthesised %d bytes for %d chars", len(audio_bytes), len(text))
        return audio_bytes
    except Exception as e:
        logger.error("gTTS failed: %s", e)
        return b""


async def _pyttsx(text: str) -> bytes:
    """
    Offline TTS using pyttsx3 — saves to a temp WAV then returns bytes.
    Lower quality but works without network.
    """
    try:
        import pyttsx3
        import tempfile
        import asyncio
        from pathlib import Path

        def _synth():
            engine = pyttsx3.init()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                path = tmp.name
            engine.save_to_file(text, path)
            engine.runAndWait()
            data = Path(path).read_bytes()
            Path(path).unlink(missing_ok=True)
            return data

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _synth)
    except Exception as e:
        logger.error("pyttsx3 failed: %s", e)
        return b""


async def _coqui(text: str) -> bytes:
    """
    Coqui TTS — high quality, runs locally.
    Install: pip install TTS
    Model is downloaded on first use (~500 MB).
    """
    try:
        from TTS.api import TTS
        import asyncio
        import tempfile
        from pathlib import Path

        def _synth():
            tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                path = tmp.name
            tts.tts_to_file(text=text, file_path=path)
            data = Path(path).read_bytes()
            Path(path).unlink(missing_ok=True)
            return data

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _synth)
    except Exception as e:
        logger.error("Coqui TTS failed: %s — falling back to gTTS", e)
        return await _gtts(text)


async def chunk_synthesise(text: str, chunk_size: int = 200) -> list[bytes]:
    """
    Split long text into chunks and synthesise each independently.
    Allows audio streaming to begin before the full response is ready.
    """
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    audio_chunks = []
    for chunk in chunks:
        audio = await synthesise(chunk)
        if audio:
            audio_chunks.append(audio)
    return audio_chunks
