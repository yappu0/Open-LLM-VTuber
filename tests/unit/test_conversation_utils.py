"""Tests for conversation utility functions."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from src.open_llm_vtuber.conversations.conversation_utils import (
    create_batch_input,
    send_conversation_start_signals,
    process_user_input,
    cleanup_conversation,
    EMOJI_LIST,
)
from src.open_llm_vtuber.agent.input_types import (
    BatchInput,
    TextData,
    ImageData,
    TextSource,
    ImageSource,
)


class TestCreateBatchInput:
    """Tests for the create_batch_input function."""

    def test_create_basic_batch_input(self):
        """Test creating a basic batch input with text only."""
        result = create_batch_input(
            input_text="Hello, world!",
            images=None,
            from_name="User",
        )

        assert isinstance(result, BatchInput)
        assert len(result.texts) == 1
        assert result.texts[0].content == "Hello, world!"
        assert result.texts[0].from_name == "User"
        assert result.texts[0].source == TextSource.INPUT
        assert result.images is None

    def test_create_batch_input_with_images(self):
        """Test creating batch input with images."""
        images = [
            {
                "source": "camera",
                "data": "base64_encoded_image_data",
                "mime_type": "image/jpeg",
            },
            {
                "source": "screen",
                "data": "another_base64_data",
                "mime_type": "image/png",
            },
        ]

        result = create_batch_input(
            input_text="What's in this image?",
            images=images,
            from_name="TestUser",
        )

        assert isinstance(result, BatchInput)
        assert len(result.texts) == 1
        assert result.images is not None
        assert len(result.images) == 2
        assert result.images[0].source == ImageSource.CAMERA
        assert result.images[0].data == "base64_encoded_image_data"
        assert result.images[0].mime_type == "image/jpeg"
        assert result.images[1].source == ImageSource.SCREEN

    def test_create_batch_input_with_metadata(self):
        """Test creating batch input with metadata."""
        metadata = {
            "proactive_speak": True,
            "skip_memory": False,
        }

        result = create_batch_input(
            input_text="Proactive message",
            images=None,
            from_name="System",
            metadata=metadata,
        )

        assert result.metadata is not None
        assert result.metadata["proactive_speak"] is True
        assert result.metadata["skip_memory"] is False

    def test_create_batch_input_empty_text(self):
        """Test creating batch input with empty text."""
        result = create_batch_input(
            input_text="",
            images=None,
            from_name="User",
        )

        assert result.texts[0].content == ""

    def test_create_batch_input_empty_images_list(self):
        """Test creating batch input with empty images list."""
        result = create_batch_input(
            input_text="No images here",
            images=[],
            from_name="User",
        )

        # Empty list should result in None for images
        assert result.images is None


class TestSendConversationStartSignals:
    """Tests for the send_conversation_start_signals function."""

    @pytest.mark.asyncio
    async def test_send_conversation_start_signals(self, mock_websocket):
        """Test sending conversation start signals."""
        send_func = mock_websocket.send_text

        await send_conversation_start_signals(send_func)

        assert mock_websocket.sent_messages is not None
        assert len(mock_websocket.sent_messages) == 2

        # Parse the messages
        messages = mock_websocket.get_sent_json_messages()

        # First message should be control
        assert messages[0]["type"] == "control"
        assert messages[0]["text"] == "conversation-chain-start"

        # Second message should be thinking indicator
        assert messages[1]["type"] == "full-text"
        assert messages[1]["text"] == "Thinking..."


class TestProcessUserInput:
    """Tests for the process_user_input function."""

    @pytest.mark.asyncio
    async def test_process_text_input(self, mock_websocket):
        """Test processing text input directly returns the text."""
        mock_asr = MagicMock()
        send_func = mock_websocket.send_text

        result = await process_user_input(
            user_input="Hello, this is text input",
            asr_engine=mock_asr,
            websocket_send=send_func,
        )

        assert result == "Hello, this is text input"
        # ASR should not be called for text input
        mock_asr.async_transcribe_np.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_audio_input(self, mock_websocket, sample_audio_np):
        """Test processing audio input calls ASR and sends transcription."""
        mock_asr = MagicMock()
        mock_asr.async_transcribe_np = AsyncMock(return_value="Transcribed text")
        send_func = mock_websocket.send_text

        result = await process_user_input(
            user_input=sample_audio_np,
            asr_engine=mock_asr,
            websocket_send=send_func,
        )

        assert result == "Transcribed text"
        mock_asr.async_transcribe_np.assert_called_once()

        # Check that transcription was sent via WebSocket
        messages = mock_websocket.get_sent_json_messages()
        assert len(messages) == 1
        assert messages[0]["type"] == "user-input-transcription"
        assert messages[0]["text"] == "Transcribed text"


class TestCleanupConversation:
    """Tests for the cleanup_conversation function."""

    def test_cleanup_clears_tts_manager(self):
        """Test that cleanup clears the TTS manager."""
        mock_tts_manager = MagicMock()

        cleanup_conversation(mock_tts_manager, "🐱")

        mock_tts_manager.clear.assert_called_once()

    def test_cleanup_with_different_emoji(self):
        """Test cleanup with different session emojis."""
        mock_tts_manager = MagicMock()

        # Test with various emojis
        for emoji in ["🐶", "🌵", "🎉"]:
            mock_tts_manager.reset_mock()
            cleanup_conversation(mock_tts_manager, emoji)
            mock_tts_manager.clear.assert_called_once()


class TestEmojiList:
    """Tests for the EMOJI_LIST constant."""

    def test_emoji_list_not_empty(self):
        """Test that emoji list is not empty."""
        assert len(EMOJI_LIST) > 0

    def test_emoji_list_contains_expected_emojis(self):
        """Test that emoji list contains expected emojis."""
        # Check for some specific emojis that should be in the list
        assert "🐶" in EMOJI_LIST
        assert "🐱" in EMOJI_LIST
        assert "🌵" in EMOJI_LIST
        assert "🎉" in EMOJI_LIST

    def test_emoji_list_is_list(self):
        """Test that EMOJI_LIST is a list."""
        assert isinstance(EMOJI_LIST, list)

    def test_emoji_list_contains_strings(self):
        """Test that all items in emoji list are strings."""
        for emoji in EMOJI_LIST:
            assert isinstance(emoji, str)


class TestBatchInputDataTypes:
    """Tests for BatchInput and related data types."""

    def test_text_data_creation(self):
        """Test creating TextData objects."""
        text_data = TextData(
            source=TextSource.INPUT,
            content="Test content",
            from_name="TestUser",
        )

        assert text_data.source == TextSource.INPUT
        assert text_data.content == "Test content"
        assert text_data.from_name == "TestUser"

    def test_text_data_default_from_name(self):
        """Test TextData with default from_name."""
        text_data = TextData(
            source=TextSource.INPUT,
            content="Test content",
        )

        assert text_data.from_name is None

    def test_image_data_creation(self):
        """Test creating ImageData objects."""
        image_data = ImageData(
            source=ImageSource.CAMERA,
            data="base64_image_data",
            mime_type="image/png",
        )

        assert image_data.source == ImageSource.CAMERA
        assert image_data.data == "base64_image_data"
        assert image_data.mime_type == "image/png"

    def test_text_source_enum_values(self):
        """Test TextSource enum values."""
        assert TextSource.INPUT.value == "input"
        assert TextSource.CLIPBOARD.value == "clipboard"

    def test_image_source_enum_values(self):
        """Test ImageSource enum values."""
        assert ImageSource.CAMERA.value == "camera"
        assert ImageSource.SCREEN.value == "screen"
        assert ImageSource.CLIPBOARD.value == "clipboard"
        assert ImageSource.UPLOAD.value == "upload"

    def test_batch_input_with_all_fields(self):
        """Test BatchInput with all fields populated."""
        texts = [
            TextData(source=TextSource.INPUT, content="Main input"),
            TextData(source=TextSource.CLIPBOARD, content="Clipboard text"),
        ]
        images = [
            ImageData(
                source=ImageSource.CAMERA,
                data="camera_data",
                mime_type="image/jpeg",
            ),
        ]
        metadata = {"test_key": "test_value"}

        batch_input = BatchInput(
            texts=texts,
            images=images,
            metadata=metadata,
        )

        assert len(batch_input.texts) == 2
        assert len(batch_input.images) == 1
        assert batch_input.metadata["test_key"] == "test_value"


class TestMockWebSocket:
    """Tests for the MockWebSocket functionality."""

    def test_mock_websocket_initialization(self, mock_websocket):
        """Test MockWebSocket initializes correctly."""
        assert mock_websocket.sent_messages == []
        assert mock_websocket.sent_bytes == []
        assert mock_websocket.receive_queue == []
        assert not mock_websocket.closed
        assert not mock_websocket.accepted

    @pytest.mark.asyncio
    async def test_mock_websocket_accept(self, mock_websocket):
        """Test MockWebSocket accept method."""
        await mock_websocket.accept()
        assert mock_websocket.accepted

    @pytest.mark.asyncio
    async def test_mock_websocket_send_text(self, mock_websocket):
        """Test MockWebSocket send_text method."""
        await mock_websocket.send_text("Hello")
        await mock_websocket.send_text("World")

        assert len(mock_websocket.sent_messages) == 2
        assert mock_websocket.sent_messages[0] == "Hello"
        assert mock_websocket.sent_messages[1] == "World"

    @pytest.mark.asyncio
    async def test_mock_websocket_send_json(self, mock_websocket):
        """Test MockWebSocket send_json method."""
        await mock_websocket.send_json({"key": "value"})

        messages = mock_websocket.get_sent_json_messages()
        assert messages[0]["key"] == "value"

    @pytest.mark.asyncio
    async def test_mock_websocket_receive_text(self, mock_websocket):
        """Test MockWebSocket receive_text method."""
        mock_websocket.queue_message("Test message")
        result = await mock_websocket.receive_text()

        assert result == "Test message"

    @pytest.mark.asyncio
    async def test_mock_websocket_close(self, mock_websocket):
        """Test MockWebSocket close method."""
        await mock_websocket.close()
        assert mock_websocket.closed

    def test_mock_websocket_clear(self, mock_websocket):
        """Test MockWebSocket clear method."""
        mock_websocket.sent_messages.append("test")
        mock_websocket.receive_queue.append("test")

        mock_websocket.clear()

        assert len(mock_websocket.sent_messages) == 0
        assert len(mock_websocket.receive_queue) == 0
