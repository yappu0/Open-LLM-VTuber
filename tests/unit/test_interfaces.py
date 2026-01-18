"""
Tests for engine interfaces (ASR, TTS, VAD).
"""

import asyncio
import os
import tempfile

import numpy as np
import pytest

from open_llm_vtuber.asr.asr_interface import ASRInterface
from open_llm_vtuber.tts.tts_interface import TTSInterface
from open_llm_vtuber.vad.vad_interface import VADInterface


# Mock implementations for testing


class MockASR(ASRInterface):
    """Mock ASR implementation for testing."""

    def __init__(self, transcription: str = "Hello, world!"):
        self.transcription = transcription
        self.call_count = 0

    def transcribe_np(self, audio: np.ndarray) -> str:
        self.call_count += 1
        return self.transcription


class MockTTS(TTSInterface):
    """Mock TTS implementation for testing."""

    def __init__(self):
        self.call_count = 0
        self.last_text = None

    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        self.call_count += 1
        self.last_text = text
        file_path = self.generate_cache_file_name(file_name_no_ext, "wav")
        # Create an empty file for testing
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(b"RIFF" + b"\x00" * 40)  # Minimal WAV header
        return file_path


class MockVAD(VADInterface):
    """Mock VAD implementation for testing."""

    def __init__(self, speech_detected: bool = True):
        self.speech_detected = speech_detected

    def detect_speech(self, audio_data: bytes):
        if self.speech_detected:
            yield audio_data


class TestASRInterface:
    """Tests for ASR interface."""

    @pytest.fixture
    def asr(self):
        """Create a MockASR instance."""
        return MockASR()

    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio data."""
        return np.random.randn(16000).astype(np.float32)

    def test_sample_rate(self):
        """Test default sample rate."""
        assert ASRInterface.SAMPLE_RATE == 16000

    def test_num_channels(self):
        """Test default number of channels."""
        assert ASRInterface.NUM_CHANNELS == 1

    def test_sample_width(self):
        """Test default sample width."""
        assert ASRInterface.SAMPLE_WIDTH == 2

    def test_transcribe_np(self, asr, sample_audio):
        """Test synchronous transcription."""
        result = asr.transcribe_np(sample_audio)
        assert result == "Hello, world!"
        assert asr.call_count == 1

    @pytest.mark.asyncio
    async def test_async_transcribe_np(self, asr, sample_audio):
        """Test async transcription wrapper."""
        result = await asr.async_transcribe_np(sample_audio)
        assert result == "Hello, world!"
        assert asr.call_count == 1

    @pytest.mark.asyncio
    async def test_async_transcribe_converts_dtype(self, asr):
        """Test that async_transcribe_np converts non-float32 arrays."""
        audio = np.random.randn(16000).astype(np.float64)
        result = await asr.async_transcribe_np(audio)
        assert result == "Hello, world!"

    def test_nparray_to_audio_file(self, asr, sample_audio):
        """Test converting numpy array to audio file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            file_path = f.name

        try:
            asr.nparray_to_audio_file(sample_audio, 16000, file_path)
            assert os.path.exists(file_path)
            assert os.path.getsize(file_path) > 0
        finally:
            os.unlink(file_path)

    def test_nparray_to_audio_file_clips_values(self, asr):
        """Test that audio values are clipped to [-1, 1]."""
        # Audio with values outside [-1, 1]
        audio = np.array([2.0, -2.0, 0.5, -0.5], dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            file_path = f.name

        try:
            asr.nparray_to_audio_file(audio, 16000, file_path)
            assert os.path.exists(file_path)
        finally:
            os.unlink(file_path)


class TestTTSInterface:
    """Tests for TTS interface."""

    @pytest.fixture
    def tts(self):
        """Create a MockTTS instance."""
        return MockTTS()

    def test_generate_audio(self, tts):
        """Test synchronous audio generation."""
        result = tts.generate_audio("Hello, world!")
        assert result.endswith(".wav")
        assert tts.call_count == 1
        assert tts.last_text == "Hello, world!"

        # Clean up
        if os.path.exists(result):
            os.unlink(result)

    @pytest.mark.asyncio
    async def test_async_generate_audio(self, tts):
        """Test async audio generation wrapper."""
        result = await tts.async_generate_audio("Test message")
        assert result.endswith(".wav")
        assert tts.call_count == 1

        # Clean up
        if os.path.exists(result):
            os.unlink(result)

    def test_generate_cache_file_name_default(self, tts):
        """Test cache file name generation with defaults."""
        result = tts.generate_cache_file_name()
        assert result == os.path.join("cache", "temp.wav")

    def test_generate_cache_file_name_custom(self, tts):
        """Test cache file name generation with custom name."""
        result = tts.generate_cache_file_name("custom_name", "mp3")
        assert result == os.path.join("cache", "custom_name.mp3")

    def test_remove_file_existing(self, tts):
        """Test removing an existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file_path = f.name

        assert os.path.exists(file_path)
        tts.remove_file(file_path)
        assert not os.path.exists(file_path)

    def test_remove_file_nonexistent(self, tts):
        """Test removing a nonexistent file doesn't raise."""
        tts.remove_file("/nonexistent/path/file.wav")
        # Should not raise


class TestVADInterface:
    """Tests for VAD interface."""

    @pytest.fixture
    def vad_with_speech(self):
        """Create a MockVAD that detects speech."""
        return MockVAD(speech_detected=True)

    @pytest.fixture
    def vad_without_speech(self):
        """Create a MockVAD that doesn't detect speech."""
        return MockVAD(speech_detected=False)

    def test_detect_speech_with_voice(self, vad_with_speech):
        """Test speech detection when voice is present."""
        audio_data = b"\x00" * 1024
        results = list(vad_with_speech.detect_speech(audio_data))
        assert len(results) == 1
        assert results[0] == audio_data

    def test_detect_speech_without_voice(self, vad_without_speech):
        """Test speech detection when no voice is present."""
        audio_data = b"\x00" * 1024
        results = list(vad_without_speech.detect_speech(audio_data))
        assert len(results) == 0


class TestInterfaceAbstraction:
    """Tests for interface abstraction properties."""

    def test_asr_is_abstract(self):
        """Test that ASRInterface requires implementation."""
        with pytest.raises(TypeError):
            # Can't instantiate abstract class
            class IncompleteASR(ASRInterface):
                pass

            IncompleteASR()

    def test_tts_is_abstract(self):
        """Test that TTSInterface requires implementation."""
        with pytest.raises(TypeError):
            # Can't instantiate abstract class
            class IncompleteTTS(TTSInterface):
                pass

            IncompleteTTS()

    def test_vad_is_abstract(self):
        """Test that VADInterface requires implementation."""
        with pytest.raises(TypeError):
            # Can't instantiate abstract class
            class IncompleteVAD(VADInterface):
                pass

            IncompleteVAD()


class TestConcurrentASR:
    """Tests for concurrent ASR operations."""

    @pytest.fixture
    def slow_asr(self):
        """Create an ASR that simulates processing time."""

        class SlowASR(ASRInterface):
            def __init__(self):
                self.call_count = 0

            def transcribe_np(self, audio: np.ndarray) -> str:
                import time

                time.sleep(0.01)  # Small delay
                self.call_count += 1
                return f"Transcription {self.call_count}"

        return SlowASR()

    @pytest.mark.asyncio
    async def test_concurrent_transcriptions(self, slow_asr):
        """Test that multiple transcriptions can run concurrently."""
        audio = np.random.randn(16000).astype(np.float32)

        # Run multiple transcriptions concurrently
        tasks = [slow_asr.async_transcribe_np(audio) for _ in range(3)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert slow_asr.call_count == 3
