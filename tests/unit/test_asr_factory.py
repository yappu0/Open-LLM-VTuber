"""Tests for ASR factory functionality."""

import pytest
from unittest.mock import patch, MagicMock

from src.open_llm_vtuber.asr.asr_factory import ASRFactory


class TestASRFactory:
    """Tests for the ASRFactory class."""

    def test_get_faster_whisper_asr(self):
        """Test creating a Faster Whisper ASR engine with mocked implementation."""
        # Skip if faster_whisper is not installed (cannot patch non-existent module)
        pytest.importorskip("faster_whisper")

        mock_asr_class = MagicMock()
        # Patch the actual module where VoiceRecognition is defined
        with patch(
            "src.open_llm_vtuber.asr.faster_whisper_asr.VoiceRecognition",
            mock_asr_class,
        ):
            result = ASRFactory.get_asr_system(
                "faster_whisper",
                model_path="large-v3",
                download_root="models",
                language="en",
                device="cpu",
                compute_type="int8",
            )
            mock_asr_class.assert_called_once_with(
                model_path="large-v3",
                download_root="models",
                language="en",
                device="cpu",
                compute_type="int8",
                prompt=None,
            )
            assert result is mock_asr_class.return_value

    def test_get_sherpa_onnx_asr(self):
        """Test creating a Sherpa ONNX ASR engine with mocked implementation."""
        # Skip if sherpa_onnx is not installed (cannot patch non-existent module)
        pytest.importorskip("sherpa_onnx")

        mock_asr_class = MagicMock()
        # Patch the actual module where VoiceRecognition is defined
        with patch(
            "src.open_llm_vtuber.asr.sherpa_onnx_asr.VoiceRecognition",
            mock_asr_class,
        ):
            result = ASRFactory.get_asr_system(
                "sherpa_onnx_asr",
                model_type="sense_voice",
                sense_voice="model.onnx",
                tokens="tokens.txt",
            )
            mock_asr_class.assert_called_once()
            assert result is mock_asr_class.return_value

    def test_unknown_asr_system_raises_error(self):
        """Test that unknown ASR system raises ValueError."""
        with pytest.raises(ValueError, match="Unknown ASR system"):
            ASRFactory.get_asr_system("nonexistent_asr")

    @pytest.mark.parametrize(
        "asr_type",
        [
            "faster_whisper",
            "whisper_cpp",
            "whisper",
            "fun_asr",
            "azure_asr",
            "groq_whisper_asr",
            "sherpa_onnx_asr",
        ],
    )
    def test_all_asr_types_recognized(self, asr_type: str):
        """Test that all ASR types are recognized by the factory.

        This test verifies that the factory code path for each ASR type exists
        and doesn't raise ValueError (which would indicate an unknown type).
        ImportError is expected when optional dependencies are not installed.
        """
        try:
            # Call the real factory - it will try to import the implementation
            ASRFactory.get_asr_system(asr_type)
        except ImportError:
            # Expected when optional dependency is not installed
            pass
        except ValueError as e:
            if "Unknown ASR system" in str(e):
                pytest.fail(f"ASR type '{asr_type}' should be recognized by factory")
            # Other ValueErrors (e.g., missing config) are acceptable
            pass
        except Exception:
            # Other errors (TypeError, etc.) are acceptable - they indicate
            # the factory recognized the type but couldn't create the instance
            pass


class TestASRFactoryWithMockedImports:
    """Tests for ASR factory with mocked module imports."""

    @pytest.mark.skip(reason="faster_whisper not installed - optional dependency")
    def test_faster_whisper_import_and_instantiation(self):
        """Test that faster_whisper module is imported and class is instantiated."""
        mock_asr_class = MagicMock()

        with patch.object(
            ASRFactory, "get_asr_system", wraps=ASRFactory.get_asr_system
        ):
            with patch(
                "src.open_llm_vtuber.asr.faster_whisper_asr.VoiceRecognition",
                mock_asr_class,
            ):
                try:
                    ASRFactory.get_asr_system(
                        "faster_whisper",
                        model_path="large-v3",
                        download_root="models",
                        language="en",
                        device="cpu",
                        compute_type="int8",
                        prompt=None,
                    )
                    mock_asr_class.assert_called_once_with(
                        model_path="large-v3",
                        download_root="models",
                        language="en",
                        device="cpu",
                        compute_type="int8",
                        prompt=None,
                    )
                except ImportError:
                    pytest.skip("Module not installed")

    @pytest.mark.skip(reason="pywhispercpp not installed - optional dependency")
    def test_whisper_cpp_import_and_instantiation(self):
        """Test that whisper_cpp module is imported and class is instantiated."""
        mock_asr_class = MagicMock()

        with patch(
            "src.open_llm_vtuber.asr.whisper_cpp_asr.VoiceRecognition",
            mock_asr_class,
        ):
            try:
                ASRFactory.get_asr_system(
                    "whisper_cpp",
                    model_name="base",
                    model_dir="models",
                )
                mock_asr_class.assert_called_once()
            except ImportError:
                pytest.skip("Module not installed")

    def test_azure_asr_import_and_instantiation(self):
        """Test that azure_asr module is imported and class is instantiated."""
        mock_asr_class = MagicMock()

        with patch(
            "src.open_llm_vtuber.asr.azure_asr.VoiceRecognition",
            mock_asr_class,
        ):
            try:
                ASRFactory.get_asr_system(
                    "azure_asr",
                    api_key="test_key",
                    region="eastus",
                    languages=["en-US"],
                )
                mock_asr_class.assert_called_once_with(
                    subscription_key="test_key",
                    region="eastus",
                    languages=["en-US"],
                )
            except ImportError:
                pytest.skip("Module not installed")

    def test_groq_whisper_asr_import_and_instantiation(self):
        """Test that groq_whisper_asr module is imported and class is instantiated."""
        mock_asr_class = MagicMock()

        with patch(
            "src.open_llm_vtuber.asr.groq_whisper_asr.VoiceRecognition",
            mock_asr_class,
        ):
            try:
                ASRFactory.get_asr_system(
                    "groq_whisper_asr",
                    api_key="test_key",
                    model="whisper-large-v3-turbo",
                    lang="en",
                )
                mock_asr_class.assert_called_once_with(
                    api_key="test_key",
                    model="whisper-large-v3-turbo",
                    lang="en",
                )
            except ImportError:
                pytest.skip("Module not installed")


class TestASRInterface:
    """Tests for ASR interface compliance."""

    def test_mock_asr_engine_implements_interface(self, mock_asr_engine):
        """Test that mock ASR engine implements the interface correctly."""
        import numpy as np

        # Test transcribe_np method
        audio = np.zeros(16000, dtype=np.float32)
        result = mock_asr_engine.transcribe_np(audio)

        assert isinstance(result, str)
        assert result == "Hello from test"
        assert mock_asr_engine.call_count == 1

    def test_mock_asr_engine_tracks_calls(self, mock_asr_engine, sample_audio_np):
        """Test that mock ASR engine tracks method calls."""
        # Multiple calls
        mock_asr_engine.transcribe_np(sample_audio_np)
        mock_asr_engine.transcribe_np(sample_audio_np)
        mock_asr_engine.transcribe_np(sample_audio_np)

        assert mock_asr_engine.call_count == 3

    def test_mock_asr_engine_stores_last_audio(self, mock_asr_engine, sample_audio_np):
        """Test that mock ASR engine stores the last audio input."""
        mock_asr_engine.transcribe_np(sample_audio_np)

        assert mock_asr_engine.last_audio is sample_audio_np
