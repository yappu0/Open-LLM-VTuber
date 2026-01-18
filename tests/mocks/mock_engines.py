"""Mock engine classes for testing."""

from typing import AsyncIterator, List, Optional
import numpy as np

from src.open_llm_vtuber.asr.asr_interface import ASRInterface
from src.open_llm_vtuber.tts.tts_interface import TTSInterface
from src.open_llm_vtuber.vad.vad_interface import VADInterface
from src.open_llm_vtuber.agent.agents.agent_interface import AgentInterface
from src.open_llm_vtuber.agent.input_types import BaseInput
from src.open_llm_vtuber.agent.output_types import (
    BaseOutput,
    SentenceOutput,
    DisplayText,
    Actions,
)


class MockASREngine(ASRInterface):
    """Mock ASR engine for testing."""

    def __init__(self, transcription: str = "Hello, world!"):
        """
        Initialize the mock ASR engine.

        Args:
            transcription: The fixed transcription to return.
        """
        self.transcription = transcription
        self.call_count = 0
        self.last_audio = None

    def transcribe_np(self, audio: np.ndarray) -> str:
        """
        Mock transcription that returns a fixed string.

        Args:
            audio: The numpy array of audio data (ignored).

        Returns:
            The pre-configured transcription string.
        """
        self.call_count += 1
        self.last_audio = audio
        return self.transcription


class MockTTSEngine(TTSInterface):
    """Mock TTS engine for testing."""

    def __init__(self, audio_path: str = "cache/mock_audio.wav"):
        """
        Initialize the mock TTS engine.

        Args:
            audio_path: The fixed audio path to return.
        """
        self.audio_path = audio_path
        self.generated_texts: List[str] = []
        self.call_count = 0

    def generate_audio(self, text: str, file_name_no_ext: Optional[str] = None) -> str:
        """
        Mock audio generation that returns a fixed path.

        Args:
            text: The text to generate audio for (stored for verification).
            file_name_no_ext: Optional filename without extension.

        Returns:
            The pre-configured audio path.
        """
        self.call_count += 1
        self.generated_texts.append(text)
        return self.audio_path


class MockVADEngine(VADInterface):
    """Mock VAD engine for testing."""

    def __init__(self, has_speech: bool = True, return_audio: bytes = b"mock_audio"):
        """
        Initialize the mock VAD engine.

        Args:
            has_speech: Whether to simulate speech detection.
            return_audio: The audio bytes to return when speech is detected.
        """
        self.has_speech = has_speech
        self.return_audio = return_audio
        self.call_count = 0
        self.last_audio_data = None

    def detect_speech(self, audio_data: bytes):
        """
        Mock speech detection.

        Note: The real VADInterface uses yield (generator), but this mock
        uses a simple return for ease of testing. For tests requiring
        generator behavior, use the MockVAD class in test_interfaces.py.

        Args:
            audio_data: Input audio data.

        Returns:
            Audio bytes if speech is detected, None otherwise.
        """
        self.call_count += 1
        self.last_audio_data = audio_data
        if self.has_speech:
            return self.return_audio
        return None


class MockAgentEngine(AgentInterface):
    """Mock agent engine for testing."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        character_name: str = "AI",
        avatar: Optional[str] = None,
    ):
        """
        Initialize the mock agent engine.

        Args:
            responses: List of responses to return in chat.
            character_name: Name of the character for display.
            avatar: Avatar path for display.
        """
        self.responses = responses or ["Hello! How can I help you?"]
        self.character_name = character_name
        self.avatar = avatar
        self.call_count = 0
        self.last_input = None
        self.interrupt_count = 0
        self.last_heard_response = None
        self.memory_load_count = 0
        self.last_conf_uid = None
        self.last_history_uid = None

    async def chat(self, input_data: BaseInput) -> AsyncIterator[BaseOutput]:
        """
        Mock chat that yields pre-configured responses.

        Args:
            input_data: The input data (stored for verification).

        Yields:
            SentenceOutput objects with the pre-configured responses.
        """
        self.call_count += 1
        self.last_input = input_data
        for response in self.responses:
            yield SentenceOutput(
                display_text=DisplayText(
                    text=response, name=self.character_name, avatar=self.avatar
                ),
                tts_text=response,
                actions=Actions(),
            )

    def handle_interrupt(self, heard_response: str) -> None:
        """
        Mock interrupt handler.

        Args:
            heard_response: The part of response heard before interruption.
        """
        self.interrupt_count += 1
        self.last_heard_response = heard_response

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        """
        Mock memory loading from history.

        Args:
            conf_uid: Configuration ID.
            history_uid: History ID.
        """
        self.memory_load_count += 1
        self.last_conf_uid = conf_uid
        self.last_history_uid = history_uid
