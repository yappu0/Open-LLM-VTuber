"""
Pytest configuration and shared fixtures for Open-LLM-VTuber tests.
"""

import asyncio
import os
import shutil
import tempfile
from typing import AsyncIterator, Generator
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# Add src to path for imports
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_chat_history_dir(temp_dir: str, monkeypatch) -> str:
    """Create a temporary chat_history directory and patch the module to use it."""
    chat_history_path = os.path.join(temp_dir, "chat_history")
    os.makedirs(chat_history_path, exist_ok=True)

    # Patch os.path.join to redirect chat_history to temp dir
    original_join = os.path.join

    def patched_join(*args):
        if args and args[0] == "chat_history":
            return original_join(chat_history_path, *args[1:])
        return original_join(*args)

    monkeypatch.setattr("os.path.join", patched_join)
    return chat_history_path


@pytest.fixture
def sample_audio_data() -> np.ndarray:
    """Generate sample audio data for testing."""
    # Generate 1 second of sine wave at 440Hz
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio


@pytest.fixture
def sample_audio_bytes(sample_audio_data: np.ndarray) -> bytes:
    """Convert sample audio data to bytes."""
    return (sample_audio_data * 32767).astype(np.int16).tobytes()


@pytest.fixture
def mock_websocket() -> MagicMock:
    """Create a mock WebSocket for testing."""
    websocket = MagicMock()
    websocket.send_text = AsyncMock()
    websocket.send_json = AsyncMock()
    websocket.receive_json = AsyncMock()
    websocket.close = AsyncMock()
    return websocket


@pytest.fixture
def mock_tts_engine() -> MagicMock:
    """Create a mock TTS engine for testing."""
    from open_llm_vtuber.tts.tts_interface import TTSInterface

    engine = MagicMock(spec=TTSInterface)
    engine.async_generate_audio = AsyncMock(return_value="/tmp/test_audio.wav")
    engine.generate_audio = MagicMock(return_value="/tmp/test_audio.wav")
    engine.remove_file = MagicMock()
    return engine


@pytest.fixture
def mock_asr_engine() -> MagicMock:
    """Create a mock ASR engine for testing."""
    from open_llm_vtuber.asr.asr_interface import ASRInterface

    engine = MagicMock(spec=ASRInterface)
    engine.async_transcribe_np = AsyncMock(return_value="Hello, world!")
    engine.transcribe_np = MagicMock(return_value="Hello, world!")
    return engine


@pytest.fixture
def mock_live2d_model() -> MagicMock:
    """Create a mock Live2D model for testing."""
    model = MagicMock()
    model.model_info = {"name": "test_model"}
    model.get_expression_list = MagicMock(return_value=["happy", "sad", "neutral"])
    return model


@pytest.fixture
def mock_websocket_send() -> AsyncMock:
    """Create a mock websocket send function."""
    return AsyncMock()
