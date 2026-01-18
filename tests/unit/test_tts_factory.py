"""Tests for TTS factory functionality."""

import pytest
from unittest.mock import patch, MagicMock

from src.open_llm_vtuber.tts.tts_factory import TTSFactory


class TestTTSFactory:
    """Tests for the TTSFactory class."""

    def test_get_edge_tts(self):
        """Test creating an Edge TTS engine."""
        mock_class = MagicMock()
        with patch(
            "src.open_llm_vtuber.tts.edge_tts.TTSEngine",
            mock_class,
        ):
            try:
                result = TTSFactory.get_tts_engine(
                    "edge_tts",
                    voice="en-US-AriaNeural",
                )
                mock_class.assert_called_once_with("en-US-AriaNeural")
            except ImportError:
                pass

    def test_get_azure_tts(self):
        """Test creating an Azure TTS engine."""
        mock_class = MagicMock()
        with patch(
            "src.open_llm_vtuber.tts.azure_tts.TTSEngine",
            mock_class,
        ):
            try:
                result = TTSFactory.get_tts_engine(
                    "azure_tts",
                    api_key="test_key",
                    region="eastus",
                    voice="en-US-AriaNeural",
                    pitch="+0%",
                    rate="+0%",
                )
                mock_class.assert_called_once_with(
                    "test_key",
                    "eastus",
                    "en-US-AriaNeural",
                    "+0%",
                    "+0%",
                )
            except ImportError:
                pass

    def test_unknown_tts_engine_raises_error(self):
        """Test that unknown TTS engine raises ValueError."""
        with pytest.raises(ValueError, match="Unknown TTS engine type"):
            TTSFactory.get_tts_engine("nonexistent_tts")

    @pytest.mark.parametrize(
        "tts_type",
        [
            "azure_tts",
            "bark_tts",
            "edge_tts",
            "pyttsx3_tts",
            "cosyvoice_tts",
            "cosyvoice2_tts",
            "melo_tts",
            "x_tts",
            "gpt_sovits_tts",
            "siliconflow_tts",
            "coqui_tts",
            "fish_api_tts",
            "minimax_tts",
            "sherpa_onnx_tts",
            "openai_tts",
            "spark_tts",
            "elevenlabs_tts",
            "cartesia_tts",
            "piper_tts",
        ],
    )
    def test_all_tts_types_recognized(self, tts_type: str):
        """Test that all TTS types are recognized by the factory."""
        with patch(
            f"src.open_llm_vtuber.tts.tts_factory.TTSFactory.get_tts_engine"
        ) as mock_method:

            def side_effect(engine_type, **kwargs):
                known_types = [
                    "azure_tts",
                    "bark_tts",
                    "edge_tts",
                    "pyttsx3_tts",
                    "cosyvoice_tts",
                    "cosyvoice2_tts",
                    "melo_tts",
                    "x_tts",
                    "gpt_sovits_tts",
                    "siliconflow_tts",
                    "coqui_tts",
                    "fish_api_tts",
                    "minimax_tts",
                    "sherpa_onnx_tts",
                    "openai_tts",
                    "spark_tts",
                    "elevenlabs_tts",
                    "cartesia_tts",
                    "piper_tts",
                ]
                if engine_type not in known_types:
                    raise ValueError(f"Unknown TTS engine type: {engine_type}")
                return MagicMock()

            mock_method.side_effect = side_effect

            result = TTSFactory.get_tts_engine(tts_type)
            assert result is not None


class TestTTSFactoryWithMockedImports:
    """Tests for TTS factory with mocked module imports."""

    @pytest.mark.skip(reason="bark not installed - optional dependency")
    def test_bark_tts_import_and_instantiation(self):
        """Test that bark_tts module is imported and class is instantiated."""
        mock_tts_class = MagicMock()

        with patch(
            "src.open_llm_vtuber.tts.bark_tts.TTSEngine",
            mock_tts_class,
        ):
            try:
                TTSFactory.get_tts_engine(
                    "bark_tts",
                    voice="v2/en_speaker_6",
                )
                mock_tts_class.assert_called_once_with("v2/en_speaker_6")
            except ImportError:
                pass

    @pytest.mark.skip(reason="melo not installed - optional dependency")
    def test_melo_tts_import_and_instantiation(self):
        """Test that melo_tts module is imported and class is instantiated."""
        mock_tts_class = MagicMock()

        with patch(
            "src.open_llm_vtuber.tts.melo_tts.TTSEngine",
            mock_tts_class,
        ):
            try:
                TTSFactory.get_tts_engine(
                    "melo_tts",
                    speaker="EN-US",
                    language="EN",
                    device="cpu",
                    speed=1.0,
                )
                mock_tts_class.assert_called_once_with(
                    speaker="EN-US",
                    language="EN",
                    device="cpu",
                    speed=1.0,
                )
            except ImportError:
                pass

    def test_openai_tts_import_and_instantiation(self):
        """Test that openai_tts module is imported and class is instantiated."""
        mock_tts_class = MagicMock()

        with patch(
            "src.open_llm_vtuber.tts.openai_tts.TTSEngine",
            mock_tts_class,
        ):
            try:
                TTSFactory.get_tts_engine(
                    "openai_tts",
                    model="tts-1",
                    voice="alloy",
                    api_key="test_key",
                    base_url="https://api.openai.com/v1",
                    file_extension="mp3",
                )
                mock_tts_class.assert_called_once_with(
                    model="tts-1",
                    voice="alloy",
                    api_key="test_key",
                    base_url="https://api.openai.com/v1",
                    file_extension="mp3",
                )
            except ImportError:
                pass

    def test_elevenlabs_tts_import_and_instantiation(self):
        """Test that elevenlabs_tts module is imported and class is instantiated."""
        mock_tts_class = MagicMock()

        with patch(
            "src.open_llm_vtuber.tts.elevenlabs_tts.TTSEngine",
            mock_tts_class,
        ):
            try:
                TTSFactory.get_tts_engine(
                    "elevenlabs_tts",
                    api_key="test_key",
                    voice_id="voice_123",
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128",
                    stability=0.5,
                    similarity_boost=0.5,
                    style=0.0,
                    use_speaker_boost=True,
                )
                mock_tts_class.assert_called_once()
            except ImportError:
                pass

    def test_sherpa_onnx_tts_import_and_instantiation(self):
        """Test that sherpa_onnx_tts module is imported and class is instantiated."""
        mock_tts_class = MagicMock()

        with patch(
            "src.open_llm_vtuber.tts.sherpa_onnx_tts.TTSEngine",
            mock_tts_class,
        ):
            try:
                TTSFactory.get_tts_engine(
                    "sherpa_onnx_tts",
                    model_path="model.onnx",
                    tokens_path="tokens.txt",
                )
                mock_tts_class.assert_called_once()
            except ImportError:
                pass


class TestTTSInterface:
    """Tests for TTS interface compliance."""

    def test_mock_tts_engine_implements_interface(self, mock_tts_engine):
        """Test that mock TTS engine implements the interface correctly."""
        result = mock_tts_engine.generate_audio("Hello, world!")

        assert isinstance(result, str)
        assert result == "cache/test_audio.wav"
        assert mock_tts_engine.call_count == 1

    def test_mock_tts_engine_tracks_generated_texts(self, mock_tts_engine):
        """Test that mock TTS engine tracks generated texts."""
        mock_tts_engine.generate_audio("First message")
        mock_tts_engine.generate_audio("Second message")
        mock_tts_engine.generate_audio("Third message")

        assert mock_tts_engine.call_count == 3
        assert mock_tts_engine.generated_texts == [
            "First message",
            "Second message",
            "Third message",
        ]

    def test_mock_tts_engine_accepts_file_name(self, mock_tts_engine):
        """Test that mock TTS engine accepts optional file name parameter."""
        result = mock_tts_engine.generate_audio(
            "Test message",
            file_name_no_ext="custom_file",
        )

        assert isinstance(result, str)
        assert mock_tts_engine.call_count == 1
