"""Unit tests for multimodal Whisper audio transcription."""

import pytest
from spectrarag.multimodal.audio_transcriber import WhisperAudioTranscriber


def test_transcribe_bytes_mock():
    transcriber = WhisperAudioTranscriber(model_size="base")
    dummy_wav_bytes = b"RIFF" + b"\x00" * 32000  # simulated 1-second 16kHz audio

    res = transcriber.transcribe_bytes(dummy_wav_bytes, filename="query.wav")
    assert res.transcript != ""
    assert res.confidence >= 0.90
    assert res.duration_sec >= 0.5
    assert len(res.timestamps) > 0
    assert "start" in res.timestamps[0]

