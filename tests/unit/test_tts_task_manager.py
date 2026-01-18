"""
Tests for TTS Task Manager (conversations/tts_manager.py).
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.open_llm_vtuber.agent.output_types import Actions, DisplayText
from src.open_llm_vtuber.conversations.tts_manager import TTSTaskManager


class TestTTSTaskManagerInit:
    """Tests for TTSTaskManager initialization."""

    def test_init(self):
        """Test TTSTaskManager initialization."""
        manager = TTSTaskManager()

        assert manager.task_list == []
        assert manager._sequence_counter == 0
        assert manager._next_sequence_to_send == 0
        assert manager._sender_task is None


class TestTTSTaskManagerClear:
    """Tests for TTSTaskManager clear method."""

    def test_clear(self):
        """Test clearing the task manager."""
        manager = TTSTaskManager()
        manager._sequence_counter = 5
        manager._next_sequence_to_send = 3
        manager.task_list = [MagicMock(), MagicMock()]

        manager.clear()

        assert manager.task_list == []
        assert manager._sequence_counter == 0
        assert manager._next_sequence_to_send == 0

    @pytest.mark.asyncio
    async def test_clear_cancels_sender_task(self):
        """Test that clear cancels the sender task."""
        manager = TTSTaskManager()

        # Create a mock task
        async def dummy_task():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                pass

        manager._sender_task = asyncio.create_task(dummy_task())

        manager.clear()

        # Give the event loop a chance to process the cancellation
        await asyncio.sleep(0.01)

        # Task should be cancelled or done
        assert manager._sender_task.cancelled() or manager._sender_task.done()


class TestTTSTaskManagerSpeak:
    """Tests for TTSTaskManager speak method."""

    @pytest.fixture
    def manager(self):
        """Create a fresh TTSTaskManager."""
        return TTSTaskManager()

    @pytest.fixture
    def mock_tts_engine(self):
        """Create a mock TTS engine."""
        engine = MagicMock()
        engine.async_generate_audio = AsyncMock(return_value="/tmp/audio.wav")
        engine.remove_file = MagicMock()
        return engine

    @pytest.fixture
    def mock_live2d_model(self):
        """Create a mock Live2D model."""
        model = MagicMock()
        model.model_info = {"name": "test"}
        return model

    @pytest.mark.asyncio
    async def test_speak_empty_text_sends_silent(self, manager, mock_tts_engine, mock_live2d_model):
        """Test that empty text sends silent payload."""
        display_text = DisplayText(text="")
        actions = Actions()
        websocket_send = AsyncMock()

        # Empty or whitespace-only text
        await manager.speak(
            tts_text="   ",  # Only whitespace
            display_text=display_text,
            actions=actions,
            live2d_model=mock_live2d_model,
            tts_engine=mock_tts_engine,
            websocket_send=websocket_send,
        )

        # Wait for queue processing
        await asyncio.sleep(0.1)

        # TTS engine should not be called for empty text
        mock_tts_engine.async_generate_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_speak_punctuation_only_sends_silent(self, manager, mock_tts_engine, mock_live2d_model):
        """Test that punctuation-only text sends silent payload."""
        display_text = DisplayText(text="...")
        websocket_send = AsyncMock()

        await manager.speak(
            tts_text="...",
            display_text=display_text,
            actions=None,
            live2d_model=mock_live2d_model,
            tts_engine=mock_tts_engine,
            websocket_send=websocket_send,
        )

        await asyncio.sleep(0.1)

        # TTS engine should not be called
        mock_tts_engine.async_generate_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_speak_increments_sequence(self, manager, mock_tts_engine, mock_live2d_model):
        """Test that speak increments sequence counter."""
        display_text = DisplayText(text="Hello")
        websocket_send = AsyncMock()

        initial_sequence = manager._sequence_counter

        with patch("src.open_llm_vtuber.conversations.tts_manager.prepare_audio_payload") as mock_prepare:
            mock_prepare.return_value = {"type": "audio", "audio": "data"}

            await manager.speak(
                tts_text="Hello",
                display_text=display_text,
                actions=None,
                live2d_model=mock_live2d_model,
                tts_engine=mock_tts_engine,
                websocket_send=websocket_send,
            )

        assert manager._sequence_counter == initial_sequence + 1

    @pytest.mark.asyncio
    async def test_speak_creates_task(self, manager, mock_tts_engine, mock_live2d_model):
        """Test that speak creates a TTS task."""
        display_text = DisplayText(text="Hello")
        websocket_send = AsyncMock()

        with patch("src.open_llm_vtuber.conversations.tts_manager.prepare_audio_payload") as mock_prepare:
            mock_prepare.return_value = {"type": "audio", "audio": "data"}

            await manager.speak(
                tts_text="Hello world",
                display_text=display_text,
                actions=Actions(),
                live2d_model=mock_live2d_model,
                tts_engine=mock_tts_engine,
                websocket_send=websocket_send,
            )

        assert len(manager.task_list) == 1

    @pytest.mark.asyncio
    async def test_speak_starts_sender_task(self, manager, mock_tts_engine, mock_live2d_model):
        """Test that speak starts the sender task if not running."""
        display_text = DisplayText(text="Hello")
        websocket_send = AsyncMock()

        assert manager._sender_task is None

        with patch("src.open_llm_vtuber.conversations.tts_manager.prepare_audio_payload") as mock_prepare:
            mock_prepare.return_value = {"type": "audio", "audio": "data"}

            await manager.speak(
                tts_text="Hello",
                display_text=display_text,
                actions=None,
                live2d_model=mock_live2d_model,
                tts_engine=mock_tts_engine,
                websocket_send=websocket_send,
            )

        assert manager._sender_task is not None


class TestPayloadOrdering:
    """Tests for payload ordering functionality."""

    @pytest.fixture
    def manager(self):
        """Create a fresh TTSTaskManager."""
        return TTSTaskManager()

    @pytest.mark.asyncio
    async def test_payloads_sent_in_order(self, manager):
        """Test that payloads are sent in sequence order."""
        sent_payloads = []

        async def mock_send(data):
            payload = json.loads(data)
            sent_payloads.append(payload)

        # Manually put payloads in queue out of order
        await manager._payload_queue.put(({"text": "first", "order": 0}, 0))
        await manager._payload_queue.put(({"text": "third", "order": 2}, 2))
        await manager._payload_queue.put(({"text": "second", "order": 1}, 1))

        # Start sender task
        sender_task = asyncio.create_task(
            manager._process_payload_queue(mock_send)
        )

        # Wait for processing
        await asyncio.sleep(0.2)

        # Cancel the sender task
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass

        # Verify order
        assert len(sent_payloads) == 3
        assert sent_payloads[0]["text"] == "first"
        assert sent_payloads[1]["text"] == "second"
        assert sent_payloads[2]["text"] == "third"

    @pytest.mark.asyncio
    async def test_silent_payload_ordering(self, manager):
        """Test that silent payloads maintain order."""
        display_text = DisplayText(text="Silent message")
        actions = Actions()

        # Queue a silent payload
        await manager._send_silent_payload(display_text, actions, 0)

        # Check queue has the payload
        assert not manager._payload_queue.empty()

        payload, sequence = await manager._payload_queue.get()
        assert sequence == 0
        assert payload["audio"] is None


class TestProcessTTS:
    """Tests for _process_tts method."""

    @pytest.fixture
    def manager(self):
        """Create a fresh TTSTaskManager."""
        return TTSTaskManager()

    @pytest.fixture
    def mock_tts_engine(self):
        """Create a mock TTS engine."""
        engine = MagicMock()
        engine.async_generate_audio = AsyncMock(return_value="/tmp/audio.wav")
        engine.remove_file = MagicMock()
        return engine

    @pytest.fixture
    def mock_live2d_model(self):
        """Create a mock Live2D model."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_process_tts_generates_audio(self, manager, mock_tts_engine, mock_live2d_model):
        """Test that _process_tts generates audio."""
        display_text = DisplayText(text="Test")
        actions = Actions()

        with patch("src.open_llm_vtuber.conversations.tts_manager.prepare_audio_payload") as mock_prepare:
            mock_prepare.return_value = {"type": "audio", "audio": "base64data"}

            await manager._process_tts(
                tts_text="Test message",
                display_text=display_text,
                actions=actions,
                live2d_model=mock_live2d_model,
                tts_engine=mock_tts_engine,
                sequence_number=0,
            )

        mock_tts_engine.async_generate_audio.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_tts_cleans_up_file(self, manager, mock_tts_engine, mock_live2d_model):
        """Test that _process_tts cleans up audio file."""
        display_text = DisplayText(text="Test")

        with patch("src.open_llm_vtuber.conversations.tts_manager.prepare_audio_payload") as mock_prepare:
            mock_prepare.return_value = {"type": "audio", "audio": "data"}

            await manager._process_tts(
                tts_text="Test",
                display_text=display_text,
                actions=None,
                live2d_model=mock_live2d_model,
                tts_engine=mock_tts_engine,
                sequence_number=0,
            )

        mock_tts_engine.remove_file.assert_called_once_with("/tmp/audio.wav")

    @pytest.mark.asyncio
    async def test_process_tts_handles_error(self, manager, mock_tts_engine, mock_live2d_model):
        """Test that _process_tts handles errors gracefully."""
        display_text = DisplayText(text="Test")

        # First call raises error, second call (for silent payload) succeeds
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("TTS error")
            return {"type": "audio", "audio": None, "volumes": [], "display_text": None, "actions": None}

        with patch("src.open_llm_vtuber.conversations.tts_manager.prepare_audio_payload") as mock_prepare:
            mock_prepare.side_effect = side_effect

            # Should not raise
            await manager._process_tts(
                tts_text="Test",
                display_text=display_text,
                actions=None,
                live2d_model=mock_live2d_model,
                tts_engine=mock_tts_engine,
                sequence_number=0,
            )

        # Should still queue a silent payload
        assert not manager._payload_queue.empty()


class TestGenerateAudio:
    """Tests for _generate_audio method."""

    @pytest.fixture
    def manager(self):
        """Create a fresh TTSTaskManager."""
        return TTSTaskManager()

    @pytest.mark.asyncio
    async def test_generate_audio_calls_tts(self, manager):
        """Test that _generate_audio calls TTS engine."""
        mock_tts = MagicMock()
        mock_tts.async_generate_audio = AsyncMock(return_value="/path/to/audio.wav")

        result = await manager._generate_audio(mock_tts, "Hello world")

        assert result == "/path/to/audio.wav"
        mock_tts.async_generate_audio.assert_called_once()

        # Check that file name contains expected format
        call_args = mock_tts.async_generate_audio.call_args
        assert call_args.kwargs["text"] == "Hello world"
        assert "file_name_no_ext" in call_args.kwargs


class TestConcurrentTTSGeneration:
    """Tests for concurrent TTS generation."""

    @pytest.fixture
    def manager(self):
        """Create a fresh TTSTaskManager."""
        return TTSTaskManager()

    @pytest.fixture
    def mock_tts_engine(self):
        """Create a mock TTS engine with delayed response."""
        engine = MagicMock()

        async def delayed_generate(*args, **kwargs):
            await asyncio.sleep(0.05)  # Small delay
            return "/tmp/audio.wav"

        engine.async_generate_audio = delayed_generate
        engine.remove_file = MagicMock()
        return engine

    @pytest.fixture
    def mock_live2d_model(self):
        """Create a mock Live2D model."""
        return MagicMock()

    @pytest.mark.asyncio
    async def test_concurrent_generation(self, manager, mock_tts_engine, mock_live2d_model):
        """Test that multiple TTS tasks can run concurrently."""
        websocket_send = AsyncMock()

        with patch("src.open_llm_vtuber.conversations.tts_manager.prepare_audio_payload") as mock_prepare:
            mock_prepare.return_value = {"type": "audio", "audio": "data"}

            # Queue multiple speak tasks
            for i in range(3):
                await manager.speak(
                    tts_text=f"Message {i}",
                    display_text=DisplayText(text=f"Message {i}"),
                    actions=None,
                    live2d_model=mock_live2d_model,
                    tts_engine=mock_tts_engine,
                    websocket_send=websocket_send,
                )

        # Should have 3 tasks queued
        assert len(manager.task_list) == 3

        # Wait for tasks to complete
        await asyncio.gather(*manager.task_list, return_exceptions=True)

        # Clean up
        manager.clear()
