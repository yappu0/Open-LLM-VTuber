"""Tests for ServiceContext functionality."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any

from src.open_llm_vtuber.service_context import ServiceContext, deep_merge


class TestServiceContextInit:
    """Tests for ServiceContext initialization."""

    def test_service_context_init_default_values(self):
        """Test ServiceContext initializes with None/default values."""
        context = ServiceContext()

        assert context.config is None
        assert context.system_config is None
        assert context.character_config is None
        assert context.live2d_model is None
        assert context.asr_engine is None
        assert context.tts_engine is None
        assert context.agent_engine is None
        assert context.vad_engine is None
        assert context.translate_engine is None
        assert context.system_prompt is None
        assert context.mcp_prompt == ""
        assert context.history_uid == ""

    def test_service_context_str_representation(self, minimal_config_dict):
        """Test ServiceContext string representation with loaded config."""
        from src.open_llm_vtuber.config_manager import validate_config

        context = ServiceContext()
        config = validate_config(minimal_config_dict)

        # Set minimal required attributes for __str__ to work
        context.system_config = config.system_config
        context.character_config = config.character_config
        context.live2d_model = None

        str_repr = str(context)

        assert "ServiceContext:" in str_repr
        assert "System Config: Loaded" in str_repr


class TestServiceContextLoadCache:
    """Tests for ServiceContext.load_cache method."""

    @pytest.mark.asyncio
    async def test_load_cache_basic(
        self,
        minimal_config_dict: Dict[str, Any],
        mock_asr_engine,
        mock_tts_engine,
        mock_vad_engine,
        mock_agent_engine,
        mock_live2d_model,
    ):
        """Test loading service context from cache."""
        from src.open_llm_vtuber.config_manager import validate_config

        config = validate_config(minimal_config_dict)
        context = ServiceContext()

        # Mock the _init_mcp_components method to avoid MCP initialization
        with patch.object(context, "_init_mcp_components", new_callable=AsyncMock):
            await context.load_cache(
                config=config,
                system_config=config.system_config,
                character_config=config.character_config,
                live2d_model=mock_live2d_model,
                asr_engine=mock_asr_engine,
                tts_engine=mock_tts_engine,
                vad_engine=mock_vad_engine,
                agent_engine=mock_agent_engine,
                translate_engine=None,
            )

        assert context.config is config
        assert context.system_config is config.system_config
        assert context.character_config is config.character_config
        assert context.asr_engine is mock_asr_engine
        assert context.tts_engine is mock_tts_engine
        assert context.vad_engine is mock_vad_engine
        assert context.agent_engine is mock_agent_engine

    @pytest.mark.asyncio
    async def test_load_cache_missing_character_config_raises(
        self,
        minimal_config_dict: Dict[str, Any],
        mock_asr_engine,
        mock_tts_engine,
        mock_vad_engine,
        mock_agent_engine,
        mock_live2d_model,
    ):
        """Test that missing character_config raises ValueError."""
        from src.open_llm_vtuber.config_manager import validate_config

        config = validate_config(minimal_config_dict)
        context = ServiceContext()

        with pytest.raises(ValueError, match="character_config cannot be None"):
            await context.load_cache(
                config=config,
                system_config=config.system_config,
                character_config=None,
                live2d_model=mock_live2d_model,
                asr_engine=mock_asr_engine,
                tts_engine=mock_tts_engine,
                vad_engine=mock_vad_engine,
                agent_engine=mock_agent_engine,
                translate_engine=None,
            )

    @pytest.mark.asyncio
    async def test_load_cache_missing_system_config_raises(
        self,
        minimal_config_dict: Dict[str, Any],
        mock_asr_engine,
        mock_tts_engine,
        mock_vad_engine,
        mock_agent_engine,
        mock_live2d_model,
    ):
        """Test that missing system_config raises ValueError."""
        from src.open_llm_vtuber.config_manager import validate_config

        config = validate_config(minimal_config_dict)
        context = ServiceContext()

        with pytest.raises(ValueError, match="system_config cannot be None"):
            await context.load_cache(
                config=config,
                system_config=None,
                character_config=config.character_config,
                live2d_model=mock_live2d_model,
                asr_engine=mock_asr_engine,
                tts_engine=mock_tts_engine,
                vad_engine=mock_vad_engine,
                agent_engine=mock_agent_engine,
                translate_engine=None,
            )


class TestServiceContextClose:
    """Tests for ServiceContext.close method."""

    @pytest.mark.asyncio
    async def test_close_cleanup_mcp_client(self):
        """Test that close() cleans up MCP client."""
        context = ServiceContext()
        mock_mcp_client = AsyncMock()
        context.mcp_client = mock_mcp_client

        await context.close()

        mock_mcp_client.aclose.assert_called_once()
        assert context.mcp_client is None

    @pytest.mark.asyncio
    async def test_close_cleanup_agent_engine(self, mock_agent_engine):
        """Test that close() cleans up agent engine if it has close method."""
        context = ServiceContext()
        mock_agent_engine.close = AsyncMock()
        context.agent_engine = mock_agent_engine

        await context.close()

        mock_agent_engine.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_resources(self):
        """Test that close() works when no resources to clean up."""
        context = ServiceContext()

        # Should not raise
        await context.close()


class TestServiceContextInitEngines:
    """Tests for individual engine initialization methods."""

    def test_init_live2d(self):
        """Test Live2D model initialization."""
        context = ServiceContext()
        mock_live2d_class = MagicMock()

        # Set up character_config

        with patch(
            "src.open_llm_vtuber.service_context.Live2dModel",
            mock_live2d_class,
        ):
            # Create a minimal character config mock
            context.character_config = MagicMock()

            context.init_live2d("test_model")

            mock_live2d_class.assert_called_once_with("test_model")

    def test_init_asr_creates_new_engine(self, minimal_asr_config: Dict[str, Any]):
        """Test ASR engine initialization creates new engine."""
        from src.open_llm_vtuber.config_manager import ASRConfig

        context = ServiceContext()
        mock_asr_factory = MagicMock()

        asr_config = ASRConfig(**minimal_asr_config)
        context.character_config = MagicMock()
        context.character_config.asr_config = None

        with patch(
            "src.open_llm_vtuber.service_context.ASRFactory.get_asr_system",
            mock_asr_factory,
        ):
            context.init_asr(asr_config)

            mock_asr_factory.assert_called_once()

    def test_init_asr_skips_if_same_config(self, minimal_asr_config: Dict[str, Any]):
        """Test ASR engine initialization skips if config unchanged."""
        from src.open_llm_vtuber.config_manager import ASRConfig

        context = ServiceContext()
        mock_asr_factory = MagicMock()

        asr_config = ASRConfig(**minimal_asr_config)
        context.asr_engine = MagicMock()
        context.character_config = MagicMock()
        context.character_config.asr_config = asr_config

        with patch(
            "src.open_llm_vtuber.service_context.ASRFactory.get_asr_system",
            mock_asr_factory,
        ):
            context.init_asr(asr_config)

            mock_asr_factory.assert_not_called()

    def test_init_tts_creates_new_engine(self, minimal_tts_config: Dict[str, Any]):
        """Test TTS engine initialization creates new engine."""
        from src.open_llm_vtuber.config_manager import TTSConfig

        context = ServiceContext()
        mock_tts_factory = MagicMock()

        tts_config = TTSConfig(**minimal_tts_config)
        context.character_config = MagicMock()
        context.character_config.tts_config = None

        with patch(
            "src.open_llm_vtuber.service_context.TTSFactory.get_tts_engine",
            mock_tts_factory,
        ):
            context.init_tts(tts_config)

            mock_tts_factory.assert_called_once()

    def test_init_vad_disabled(self):
        """Test VAD initialization when disabled."""
        from src.open_llm_vtuber.config_manager import VADConfig

        context = ServiceContext()
        vad_config = VADConfig(vad_model=None)

        context.init_vad(vad_config)

        assert context.vad_engine is None


class TestDeepMerge:
    """Tests for the deep_merge utility function."""

    def test_deep_merge_simple_dicts(self):
        """Test merging simple dictionaries."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3}
        result = deep_merge(dict1, dict2)

        assert result == {"a": 1, "b": 2, "c": 3}

    def test_deep_merge_overwrites_values(self):
        """Test that dict2 values overwrite dict1 values."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 3}
        result = deep_merge(dict1, dict2)

        assert result == {"a": 1, "b": 3}

    def test_deep_merge_nested_dicts(self):
        """Test merging nested dictionaries."""
        dict1 = {
            "outer": {
                "inner1": 1,
                "inner2": 2,
            }
        }
        dict2 = {
            "outer": {
                "inner2": 3,
                "inner3": 4,
            }
        }
        result = deep_merge(dict1, dict2)

        assert result == {
            "outer": {
                "inner1": 1,
                "inner2": 3,
                "inner3": 4,
            }
        }

    def test_deep_merge_preserves_original(self):
        """Test that original dictionaries are not modified."""
        dict1 = {"a": 1, "b": {"c": 2}}
        dict2 = {"b": {"d": 3}}
        original_dict1 = {"a": 1, "b": {"c": 2}}

        deep_merge(dict1, dict2)

        assert dict1 == original_dict1

    def test_deep_merge_empty_dict2(self):
        """Test merging with empty dict2."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {}
        result = deep_merge(dict1, dict2)

        assert result == {"a": 1, "b": 2}

    def test_deep_merge_empty_dict1(self):
        """Test merging with empty dict1."""
        dict1 = {}
        dict2 = {"a": 1, "b": 2}
        result = deep_merge(dict1, dict2)

        assert result == {"a": 1, "b": 2}

    def test_deep_merge_deeply_nested(self):
        """Test merging deeply nested dictionaries."""
        dict1 = {
            "level1": {
                "level2": {
                    "level3": {
                        "value1": "a",
                        "value2": "b",
                    }
                }
            }
        }
        dict2 = {
            "level1": {
                "level2": {
                    "level3": {
                        "value2": "c",
                        "value3": "d",
                    }
                }
            }
        }
        result = deep_merge(dict1, dict2)

        assert result["level1"]["level2"]["level3"] == {
            "value1": "a",
            "value2": "c",
            "value3": "d",
        }

    def test_deep_merge_list_values_not_merged(self):
        """Test that list values are replaced, not merged."""
        dict1 = {"items": [1, 2, 3]}
        dict2 = {"items": [4, 5]}
        result = deep_merge(dict1, dict2)

        assert result == {"items": [4, 5]}

    def test_deep_merge_mixed_types(self):
        """Test merging with mixed value types."""
        dict1 = {"a": {"nested": 1}}
        dict2 = {"a": "string_value"}  # Dict replaced by string
        result = deep_merge(dict1, dict2)

        assert result == {"a": "string_value"}
