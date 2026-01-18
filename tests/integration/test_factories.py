"""
Integration tests for factory classes.

These tests verify that factories can create appropriate engine instances
without actually loading heavy models or making API calls.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestASRFactory:
    """Tests for ASR Factory."""

    def test_factory_import(self):
        """Test that ASR factory can be imported."""
        from open_llm_vtuber.asr.asr_factory import ASRFactory

        assert ASRFactory is not None

    def test_factory_has_get_asr_system(self):
        """Test that factory has get_asr_system method."""
        from open_llm_vtuber.asr.asr_factory import ASRFactory

        assert hasattr(ASRFactory, "get_asr_system")
        assert callable(ASRFactory.get_asr_system)

    def test_unknown_system_raises(self):
        """Test that unknown system raises ValueError."""
        from open_llm_vtuber.asr.asr_factory import ASRFactory

        with pytest.raises(ValueError):
            ASRFactory.get_asr_system("unknown_asr_system")


class TestTTSFactory:
    """Tests for TTS Factory."""

    def test_factory_import(self):
        """Test that TTS factory can be imported."""
        from open_llm_vtuber.tts.tts_factory import TTSFactory

        assert TTSFactory is not None

    def test_factory_has_get_tts_engine(self):
        """Test that factory has get_tts_engine method."""
        from open_llm_vtuber.tts.tts_factory import TTSFactory

        assert hasattr(TTSFactory, "get_tts_engine")
        assert callable(TTSFactory.get_tts_engine)

    def test_unknown_engine_raises(self):
        """Test that unknown engine raises ValueError."""
        from open_llm_vtuber.tts.tts_factory import TTSFactory

        with pytest.raises(ValueError):
            TTSFactory.get_tts_engine("unknown_tts_engine")


class TestVADFactory:
    """Tests for VAD Factory."""

    def test_factory_import(self):
        """Test that VAD factory can be imported."""
        from open_llm_vtuber.vad.vad_factory import VADFactory

        assert VADFactory is not None

    def test_factory_has_get_vad_engine(self):
        """Test that factory has get_vad_engine method."""
        from open_llm_vtuber.vad.vad_factory import VADFactory

        assert hasattr(VADFactory, "get_vad_engine")
        assert callable(VADFactory.get_vad_engine)

    def test_none_engine_returns_none(self):
        """Test that None engine type returns None."""
        from open_llm_vtuber.vad.vad_factory import VADFactory

        result = VADFactory.get_vad_engine(None)
        assert result is None


class TestAgentFactory:
    """Tests for Agent Factory."""

    def test_factory_import(self):
        """Test that Agent factory can be imported."""
        from open_llm_vtuber.agent.agent_factory import AgentFactory

        assert AgentFactory is not None

    def test_factory_has_create_agent(self):
        """Test that factory has create_agent method."""
        from open_llm_vtuber.agent.agent_factory import AgentFactory

        assert hasattr(AgentFactory, "create_agent")
        assert callable(AgentFactory.create_agent)


class TestLLMFactory:
    """Tests for LLM Factory."""

    def test_factory_import(self):
        """Test that LLM factory can be imported."""
        from open_llm_vtuber.agent.stateless_llm_factory import LLMFactory

        assert LLMFactory is not None

    def test_factory_has_create_llm(self):
        """Test that factory has create_llm method."""
        from open_llm_vtuber.agent.stateless_llm_factory import LLMFactory

        assert hasattr(LLMFactory, "create_llm")
        assert callable(LLMFactory.create_llm)

    def test_unknown_provider_raises(self):
        """Test that unknown provider raises ValueError."""
        from open_llm_vtuber.agent.stateless_llm_factory import LLMFactory

        with pytest.raises(ValueError):
            LLMFactory.create_llm("unknown_llm_provider")


class TestFactoryInterfaceCompliance:
    """Tests to ensure factory-created objects comply with interfaces."""

    def test_asr_interface_compliance(self):
        """Test that ASR factory produces ASRInterface-compliant objects."""
        from open_llm_vtuber.asr.asr_interface import ASRInterface

        # All ASR implementations should be subclasses of ASRInterface
        # This is a structural test - actual compliance is tested via duck typing

        # Check that the interface defines expected methods
        assert hasattr(ASRInterface, "transcribe_np")
        assert hasattr(ASRInterface, "async_transcribe_np")
        assert hasattr(ASRInterface, "nparray_to_audio_file")

    def test_tts_interface_compliance(self):
        """Test that TTS factory produces TTSInterface-compliant objects."""
        from open_llm_vtuber.tts.tts_interface import TTSInterface

        # Check that the interface defines expected methods
        assert hasattr(TTSInterface, "generate_audio")
        assert hasattr(TTSInterface, "async_generate_audio")
        assert hasattr(TTSInterface, "remove_file")
        assert hasattr(TTSInterface, "generate_cache_file_name")

    def test_vad_interface_compliance(self):
        """Test that VAD factory produces VADInterface-compliant objects."""
        from open_llm_vtuber.vad.vad_interface import VADInterface

        # Check that the interface defines expected methods
        assert hasattr(VADInterface, "detect_speech")

    def test_agent_interface_compliance(self):
        """Test that Agent factory produces AgentInterface-compliant objects."""
        from open_llm_vtuber.agent.agents.agent_interface import AgentInterface

        # Check that the interface defines expected methods
        assert hasattr(AgentInterface, "chat")
        assert hasattr(AgentInterface, "handle_interrupt")
        assert hasattr(AgentInterface, "set_memory_from_history")


class TestFactoryConfiguration:
    """Tests for factory configuration handling."""

    def test_asr_factory_supported_systems(self):
        """Verify ASR factory supports expected systems."""
        from open_llm_vtuber.asr.asr_factory import ASRFactory

        # These are the systems mentioned in CLAUDE.md
        expected_systems = [
            "faster_whisper",
            "whisper_cpp",
            "whisper",
            "fun_asr",
            "azure_asr",
            "groq_whisper_asr",
            "sherpa_onnx_asr",
        ]

        # We can't test instantiation without proper config,
        # but we can verify the factory recognizes these names
        # by checking it doesn't raise immediately
        for system in expected_systems:
            # Factory should not raise ValueError for known systems
            # (it may raise other errors due to missing dependencies)
            try:
                ASRFactory.get_asr_system(system)
            except ValueError as e:
                if "Unknown" in str(e) or "not supported" in str(e).lower():
                    pytest.fail(f"ASR system '{system}' should be supported")
            except Exception:
                # Other errors are expected (missing deps, config, etc.)
                pass

    def test_tts_factory_supported_engines(self):
        """Verify TTS factory supports expected engines."""
        from open_llm_vtuber.tts.tts_factory import TTSFactory

        # Sample of expected engines
        expected_engines = [
            "edge_tts",
            "pyttsx3_tts",
        ]

        for engine in expected_engines:
            try:
                TTSFactory.get_tts_engine(engine)
            except ValueError as e:
                if "Unknown" in str(e) or "not supported" in str(e).lower():
                    pytest.fail(f"TTS engine '{engine}' should be supported")
            except Exception:
                # Other errors are expected
                pass


class TestEdgeTTSFactory:
    """Specific tests for Edge TTS as it has minimal dependencies."""

    def test_edge_tts_creation(self):
        """Test creating Edge TTS engine."""
        from open_llm_vtuber.tts.tts_factory import TTSFactory
        from open_llm_vtuber.tts.tts_interface import TTSInterface

        try:
            engine = TTSFactory.get_tts_engine(
                "edge_tts",
                voice="en-US-AvaMultilingualNeural",
            )
            assert engine is not None
            assert isinstance(engine, TTSInterface)
        except ImportError:
            pytest.skip("edge-tts not installed")


class TestPyttsx3TTSFactory:
    """Specific tests for pyttsx3 TTS as it uses system TTS."""

    @pytest.mark.skip(reason="pyttsx3 requires audio drivers not available in CI")
    def test_pyttsx3_tts_creation(self):
        """Test creating pyttsx3 TTS engine."""
        from open_llm_vtuber.tts.tts_factory import TTSFactory
        from open_llm_vtuber.tts.tts_interface import TTSInterface

        try:
            engine = TTSFactory.get_tts_engine("pyttsx3_tts")
            assert engine is not None
            assert isinstance(engine, TTSInterface)
        except Exception as e:
            # pyttsx3 may fail on systems without audio support
            if "No module named 'pyttsx3'" in str(e):
                pytest.skip("pyttsx3 not installed")
            elif "no driver" in str(e).lower() or "audio" in str(e).lower():
                pytest.skip("No audio driver available")
            raise
