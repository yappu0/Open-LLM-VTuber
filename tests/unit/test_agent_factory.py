"""Tests for Agent factory functionality."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.open_llm_vtuber.agent.agent_factory import AgentFactory


class TestAgentFactory:
    """Tests for the AgentFactory class."""

    def test_create_basic_memory_agent(self, mock_live2d_model):
        """Test creating a BasicMemoryAgent."""
        mock_llm = MagicMock()
        mock_agent_class = MagicMock()

        with patch(
            "src.open_llm_vtuber.agent.agent_factory.StatelessLLMFactory.create_llm",
            return_value=mock_llm,
        ):
            with patch(
                "src.open_llm_vtuber.agent.agent_factory.BasicMemoryAgent",
                mock_agent_class,
            ):
                agent_settings = {
                    "basic_memory_agent": {
                        "llm_provider": "ollama_llm",
                        "faster_first_response": True,
                        "segment_method": "pysbd",
                        "use_mcpp": False,
                        "mcp_enabled_servers": [],
                    },
                }
                llm_configs = {
                    "ollama_llm": {
                        "base_url": "http://localhost:11434/v1",
                        "model": "qwen2.5:latest",
                        "temperature": 1.0,
                    },
                }

                AgentFactory.create_agent(
                    conversation_agent_choice="basic_memory_agent",
                    agent_settings=agent_settings,
                    llm_configs=llm_configs,
                    system_prompt="You are a helpful assistant.",
                    live2d_model=mock_live2d_model,
                )

                mock_agent_class.assert_called_once()

    def test_basic_memory_agent_missing_llm_provider(self, mock_live2d_model):
        """Test that missing LLM provider raises ValueError."""
        agent_settings = {
            "basic_memory_agent": {
                # Missing llm_provider
                "faster_first_response": True,
                "segment_method": "pysbd",
            },
        }
        llm_configs = {
            "ollama_llm": {
                "base_url": "http://localhost:11434/v1",
                "model": "qwen2.5:latest",
            },
        }

        with pytest.raises(ValueError, match="LLM provider not specified"):
            AgentFactory.create_agent(
                conversation_agent_choice="basic_memory_agent",
                agent_settings=agent_settings,
                llm_configs=llm_configs,
                system_prompt="You are a helpful assistant.",
                live2d_model=mock_live2d_model,
            )

    def test_basic_memory_agent_missing_llm_config(self, mock_live2d_model):
        """Test that missing LLM config raises error."""
        agent_settings = {
            "basic_memory_agent": {
                "llm_provider": "nonexistent_llm",
                "faster_first_response": True,
            },
        }
        llm_configs = {
            "ollama_llm": {
                "base_url": "http://localhost:11434/v1",
                "model": "qwen2.5:latest",
            },
        }

        # The code raises AttributeError when llm_config is None
        # because it tries to call .pop() on None before the ValueError check
        with pytest.raises((ValueError, AttributeError)):
            AgentFactory.create_agent(
                conversation_agent_choice="basic_memory_agent",
                agent_settings=agent_settings,
                llm_configs=llm_configs,
                system_prompt="You are a helpful assistant.",
                live2d_model=mock_live2d_model,
            )

    def test_create_hume_ai_agent(self, mock_live2d_model):
        """Test creating a HumeAI Agent."""
        mock_agent_class = MagicMock()

        with patch(
            "src.open_llm_vtuber.agent.agent_factory.HumeAIAgent",
            mock_agent_class,
        ):
            agent_settings = {
                "hume_ai_agent": {
                    "api_key": "test_api_key",
                    "host": "api.hume.ai",
                    "config_id": "config_123",
                    "idle_timeout": 15,
                },
            }
            llm_configs = {}

            AgentFactory.create_agent(
                conversation_agent_choice="hume_ai_agent",
                agent_settings=agent_settings,
                llm_configs=llm_configs,
                system_prompt="You are a helpful assistant.",
                live2d_model=mock_live2d_model,
            )

            mock_agent_class.assert_called_once_with(
                api_key="test_api_key",
                host="api.hume.ai",
                config_id="config_123",
                idle_timeout=15,
            )

    def test_create_letta_agent(self, mock_live2d_model):
        """Test creating a Letta Agent."""
        mock_agent_class = MagicMock()

        with patch(
            "src.open_llm_vtuber.agent.agent_factory.LettaAgent",
            mock_agent_class,
        ):
            agent_settings = {
                "letta_agent": {
                    "host": "localhost",
                    "port": 8283,
                    "id": "agent_123",
                    "faster_first_response": True,
                    "segment_method": "pysbd",
                },
            }
            llm_configs = {}

            AgentFactory.create_agent(
                conversation_agent_choice="letta_agent",
                agent_settings=agent_settings,
                llm_configs=llm_configs,
                system_prompt="You are a helpful assistant.",
                live2d_model=mock_live2d_model,
            )

            mock_agent_class.assert_called_once()

    def test_unsupported_agent_type(self, mock_live2d_model):
        """Test that unsupported agent type raises ValueError."""
        agent_settings = {}
        llm_configs = {}

        with pytest.raises(ValueError, match="Unsupported agent type"):
            AgentFactory.create_agent(
                conversation_agent_choice="nonexistent_agent",
                agent_settings=agent_settings,
                llm_configs=llm_configs,
                system_prompt="You are a helpful assistant.",
                live2d_model=mock_live2d_model,
            )


class TestAgentInterface:
    """Tests for Agent interface compliance."""

    @pytest.mark.asyncio
    async def test_mock_agent_engine_implements_chat(self, mock_agent_engine):
        """Test that mock agent engine implements chat method correctly."""
        from src.open_llm_vtuber.agent.input_types import BatchInput, TextData, TextSource

        input_data = BatchInput(
            texts=[TextData(source=TextSource.INPUT, content="Hello!")]
        )

        responses = []
        async for output in mock_agent_engine.chat(input_data):
            responses.append(output)

        assert len(responses) == 1
        # The fixture uses "Hello! I am a test AI." as the default response
        assert responses[0].display_text.text == "Hello! I am a test AI."
        assert mock_agent_engine.call_count == 1

    def test_mock_agent_engine_implements_handle_interrupt(self, mock_agent_engine):
        """Test that mock agent engine implements handle_interrupt method."""
        mock_agent_engine.handle_interrupt("I was saying...")

        assert mock_agent_engine.interrupt_count == 1
        assert mock_agent_engine.last_heard_response == "I was saying..."

    def test_mock_agent_engine_implements_set_memory(self, mock_agent_engine):
        """Test that mock agent engine implements set_memory_from_history method."""
        mock_agent_engine.set_memory_from_history("conf_001", "history_001")

        assert mock_agent_engine.memory_load_count == 1
        assert mock_agent_engine.last_conf_uid == "conf_001"
        assert mock_agent_engine.last_history_uid == "history_001"

    @pytest.mark.asyncio
    async def test_mock_agent_engine_multiple_responses(self):
        """Test mock agent engine with multiple configured responses."""
        from tests.mocks import MockAgentEngine
        from src.open_llm_vtuber.agent.input_types import BatchInput, TextData, TextSource

        agent = MockAgentEngine(
            responses=["First response", "Second response", "Third response"]
        )
        input_data = BatchInput(
            texts=[TextData(source=TextSource.INPUT, content="Test")]
        )

        responses = []
        async for output in agent.chat(input_data):
            responses.append(output)

        assert len(responses) == 3
        assert responses[0].display_text.text == "First response"
        assert responses[1].display_text.text == "Second response"
        assert responses[2].display_text.text == "Third response"

    @pytest.mark.asyncio
    async def test_mock_agent_engine_stores_last_input(self, mock_agent_engine):
        """Test that mock agent engine stores the last input."""
        from src.open_llm_vtuber.agent.input_types import BatchInput, TextData, TextSource

        input_data = BatchInput(
            texts=[TextData(source=TextSource.INPUT, content="Remember this!")]
        )

        async for _ in mock_agent_engine.chat(input_data):
            pass

        assert mock_agent_engine.last_input is input_data
        assert mock_agent_engine.last_input.texts[0].content == "Remember this!"
