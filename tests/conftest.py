"""Shared fixtures for Open-LLM-VTuber tests."""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any

from tests.mocks import (
    MockASREngine,
    MockTTSEngine,
    MockAgentEngine,
    MockVADEngine,
    MockWebSocket,
)


# ============ Path Fixtures ============


@pytest.fixture
def tests_dir() -> Path:
    """Return the tests directory path."""
    return Path(__file__).parent


@pytest.fixture
def fixtures_dir(tests_dir: Path) -> Path:
    """Return the fixtures directory path."""
    return tests_dir / "fixtures"


@pytest.fixture
def configs_dir(fixtures_dir: Path) -> Path:
    """Return the configs fixtures directory path."""
    return fixtures_dir / "configs"


# ============ Config Fixtures ============


@pytest.fixture
def minimal_system_config() -> Dict[str, Any]:
    """Return a minimal valid system configuration dictionary."""
    return {
        "conf_version": "v1.2.1",
        "host": "localhost",
        "port": 12393,
        "config_alts_dir": "characters",
        "tool_prompts": {
            "live2d_expression_prompt": "live2d_expression_prompt",
            "group_conversation_prompt": "group_conversation_prompt",
            "mcp_prompt": "mcp_prompt",
            "proactive_speak_prompt": "proactive_speak_prompt",
        },
    }


@pytest.fixture
def minimal_asr_config() -> Dict[str, Any]:
    """Return a minimal valid ASR configuration dictionary."""
    return {
        "asr_model": "sherpa_onnx_asr",
        "sherpa_onnx_asr": {
            "model_type": "sense_voice",
            "sense_voice": "model.onnx",
            "tokens": "tokens.txt",
            "num_threads": 4,
            "use_itn": True,
            "provider": "cpu",
        },
    }


@pytest.fixture
def minimal_tts_config() -> Dict[str, Any]:
    """Return a minimal valid TTS configuration dictionary."""
    return {
        "tts_model": "edge_tts",
        "edge_tts": {
            "voice": "en-US-AriaNeural",
        },
    }


@pytest.fixture
def minimal_vad_config() -> Dict[str, Any]:
    """Return a minimal valid VAD configuration dictionary."""
    return {
        "vad_model": "silero_vad",
        "silero_vad": {
            "orig_sr": 16000,
            "target_sr": 16000,
            "prob_threshold": 0.4,
            "db_threshold": 60,
            "required_hits": 3,
            "required_misses": 24,
            "smoothing_window": 5,
        },
    }


@pytest.fixture
def minimal_agent_config() -> Dict[str, Any]:
    """Return a minimal valid agent configuration dictionary."""
    return {
        "conversation_agent_choice": "basic_memory_agent",
        "agent_settings": {
            "basic_memory_agent": {
                "llm_provider": "ollama_llm",
                "faster_first_response": True,
                "segment_method": "pysbd",
                "use_mcpp": False,
                "mcp_enabled_servers": [],
            },
        },
        "llm_configs": {
            "ollama_llm": {
                "base_url": "http://localhost:11434/v1",
                "model": "qwen2.5:latest",
                "temperature": 1.0,
                "keep_alive": -1,
                "unload_at_exit": True,
            },
        },
    }


@pytest.fixture
def minimal_tts_preprocessor_config() -> Dict[str, Any]:
    """Return a minimal valid TTS preprocessor configuration dictionary."""
    return {
        "remove_special_char": True,
        "ignore_brackets": True,
        "ignore_parentheses": True,
        "ignore_asterisks": True,
        "ignore_angle_brackets": True,
        "translator_config": {
            "translate_audio": False,
            "translate_provider": "deeplx",
            "deeplx": {
                "deeplx_target_lang": "EN",
                "deeplx_api_endpoint": "http://localhost:1188/translate",
            },
        },
    }


@pytest.fixture
def minimal_character_config(
    minimal_asr_config: Dict[str, Any],
    minimal_tts_config: Dict[str, Any],
    minimal_vad_config: Dict[str, Any],
    minimal_agent_config: Dict[str, Any],
    minimal_tts_preprocessor_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a minimal valid character configuration dictionary."""
    return {
        "conf_name": "test_character",
        "conf_uid": "test_character_001",
        "live2d_model_name": "test_model",
        "character_name": "Test AI",
        "human_name": "Human",
        "avatar": "",
        "persona_prompt": "You are a helpful AI assistant for testing purposes.",
        "agent_config": minimal_agent_config,
        "asr_config": minimal_asr_config,
        "tts_config": minimal_tts_config,
        "vad_config": minimal_vad_config,
        "tts_preprocessor_config": minimal_tts_preprocessor_config,
    }


@pytest.fixture
def minimal_config_dict(
    minimal_system_config: Dict[str, Any],
    minimal_character_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a minimal valid complete configuration dictionary."""
    return {
        "system_config": minimal_system_config,
        "character_config": minimal_character_config,
    }


@pytest.fixture
def full_config_dict(minimal_config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return a full configuration dictionary with all optional fields."""
    config = minimal_config_dict.copy()
    config["live_config"] = {
        "bilibili_live": {
            "enable_bilibili_live": False,
        },
    }
    return config


# ============ Mock Engine Fixtures ============


@pytest.fixture
def mock_asr_engine() -> MockASREngine:
    """Return a mock ASR engine instance."""
    return MockASREngine(transcription="Hello from test")


@pytest.fixture
def mock_tts_engine() -> MockTTSEngine:
    """Return a mock TTS engine instance."""
    return MockTTSEngine(audio_path="cache/test_audio.wav")


@pytest.fixture
def mock_vad_engine() -> MockVADEngine:
    """Return a mock VAD engine instance."""
    return MockVADEngine(has_speech=True)


@pytest.fixture
def mock_agent_engine() -> MockAgentEngine:
    """Return a mock agent engine instance."""
    return MockAgentEngine(
        responses=["Hello! I am a test AI."],
        character_name="Test AI",
    )


# ============ WebSocket Fixtures ============


@pytest.fixture
def mock_websocket() -> MockWebSocket:
    """Return a mock WebSocket instance."""
    return MockWebSocket()


# ============ Audio Fixtures ============


@pytest.fixture
def sample_audio_np() -> np.ndarray:
    """Return a sample numpy audio array for testing."""
    # Generate 1 second of silence at 16kHz sample rate
    duration = 1.0
    sample_rate = 16000
    samples = int(duration * sample_rate)
    return np.zeros(samples, dtype=np.float32)


@pytest.fixture
def sample_audio_with_noise() -> np.ndarray:
    """Return a sample numpy audio array with random noise for testing."""
    duration = 1.0
    sample_rate = 16000
    samples = int(duration * sample_rate)
    # Generate random noise in range [-0.1, 0.1]
    return np.random.uniform(-0.1, 0.1, samples).astype(np.float32)


@pytest.fixture
def sample_audio_bytes() -> bytes:
    """Return sample audio bytes for testing."""
    # Generate 1 second of silence at 16kHz sample rate (16-bit PCM)
    duration = 1.0
    sample_rate = 16000
    samples = int(duration * sample_rate)
    audio = np.zeros(samples, dtype=np.int16)
    return audio.tobytes()


# ============ Live2D Mock Fixtures ============


class MockLive2DModel:
    """Mock Live2D model for testing."""

    def __init__(self, model_name: str = "test_model"):
        self.model_name = model_name
        self.model_info = {
            "name": model_name,
            "url": f"/live2d-models/{model_name}/{model_name}.model3.json",
        }
        self.emo_str = "happy, sad, angry, surprised, neutral"
        self.expressions = ["happy", "sad", "angry", "surprised", "neutral"]

    def get_expression(self, expression_name: str) -> int:
        """Get expression index by name."""
        try:
            return self.expressions.index(expression_name)
        except ValueError:
            return 0


@pytest.fixture
def mock_live2d_model() -> MockLive2DModel:
    """Return a mock Live2D model instance."""
    return MockLive2DModel()
