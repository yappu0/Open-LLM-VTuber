"""
Tests for output types (output_types.py).
"""

import pytest

from open_llm_vtuber.agent.output_types import (
    Actions,
    AudioOutput,
    DisplayText,
    SentenceOutput,
)


class TestActions:
    """Tests for Actions dataclass."""

    def test_actions_default(self):
        """Test Actions with default values."""
        actions = Actions()
        assert actions.expressions is None
        assert actions.pictures is None
        assert actions.sounds is None

    def test_actions_with_expressions_strings(self):
        """Test Actions with string expressions."""
        actions = Actions(expressions=["happy", "excited"])
        assert actions.expressions == ["happy", "excited"]

    def test_actions_with_expressions_ints(self):
        """Test Actions with integer expressions (indices)."""
        actions = Actions(expressions=[0, 1, 2])
        assert actions.expressions == [0, 1, 2]

    def test_actions_with_pictures(self):
        """Test Actions with pictures."""
        actions = Actions(pictures=["image1.png", "image2.jpg"])
        assert actions.pictures == ["image1.png", "image2.jpg"]

    def test_actions_with_sounds(self):
        """Test Actions with sounds."""
        actions = Actions(sounds=["sound1.wav", "sound2.mp3"])
        assert actions.sounds == ["sound1.wav", "sound2.mp3"]

    def test_actions_to_dict_empty(self):
        """Test to_dict with empty Actions."""
        actions = Actions()
        result = actions.to_dict()
        assert result == {}

    def test_actions_to_dict_with_expressions(self):
        """Test to_dict with expressions."""
        actions = Actions(expressions=["happy"])
        result = actions.to_dict()
        assert result == {"expressions": ["happy"]}

    def test_actions_to_dict_full(self):
        """Test to_dict with all fields."""
        actions = Actions(
            expressions=["happy"],
            pictures=["pic.png"],
            sounds=["sound.wav"],
        )
        result = actions.to_dict()
        assert result == {
            "expressions": ["happy"],
            "pictures": ["pic.png"],
            "sounds": ["sound.wav"],
        }

    def test_actions_to_dict_excludes_none(self):
        """Test that to_dict excludes None values."""
        actions = Actions(expressions=["happy"], pictures=None, sounds=None)
        result = actions.to_dict()
        assert "pictures" not in result
        assert "sounds" not in result


class TestDisplayText:
    """Tests for DisplayText dataclass."""

    def test_display_text_minimal(self):
        """Test DisplayText with minimal fields."""
        display_text = DisplayText(text="Hello!")
        assert display_text.text == "Hello!"
        assert display_text.name == "AI"  # Default value
        assert display_text.avatar is None

    def test_display_text_with_name(self):
        """Test DisplayText with custom name."""
        display_text = DisplayText(text="Hi there!", name="Assistant")
        assert display_text.text == "Hi there!"
        assert display_text.name == "Assistant"

    def test_display_text_with_avatar(self):
        """Test DisplayText with avatar."""
        display_text = DisplayText(
            text="Hello!",
            name="Bot",
            avatar="/avatars/bot.png",
        )
        assert display_text.avatar == "/avatars/bot.png"

    def test_display_text_to_dict(self):
        """Test DisplayText to_dict method."""
        display_text = DisplayText(
            text="Test message",
            name="AI",
            avatar="/avatar.png",
        )
        result = display_text.to_dict()
        assert result == {
            "text": "Test message",
            "name": "AI",
            "avatar": "/avatar.png",
        }

    def test_display_text_to_dict_with_none_avatar(self):
        """Test DisplayText to_dict with None avatar."""
        display_text = DisplayText(text="Test", name="AI")
        result = display_text.to_dict()
        assert result["avatar"] is None

    def test_display_text_str(self):
        """Test DisplayText string representation."""
        display_text = DisplayText(text="Hello, world!", name="Bot")
        result = str(display_text)
        assert result == "Bot: Hello, world!"

    def test_display_text_str_default_name(self):
        """Test DisplayText string with default name."""
        display_text = DisplayText(text="Message")
        result = str(display_text)
        assert result == "AI: Message"


class TestSentenceOutput:
    """Tests for SentenceOutput dataclass."""

    def test_sentence_output_creation(self):
        """Test SentenceOutput creation."""
        display_text = DisplayText(text="Hello!")
        actions = Actions(expressions=["happy"])
        output = SentenceOutput(
            display_text=display_text,
            tts_text="Hello!",
            actions=actions,
        )
        assert output.display_text.text == "Hello!"
        assert output.tts_text == "Hello!"
        assert output.actions.expressions == ["happy"]

    def test_sentence_output_different_display_and_tts(self):
        """Test SentenceOutput with different display and TTS text."""
        display_text = DisplayText(text="Hello! :)")
        output = SentenceOutput(
            display_text=display_text,
            tts_text="Hello!",  # TTS without emoji
            actions=Actions(),
        )
        assert output.display_text.text == "Hello! :)"
        assert output.tts_text == "Hello!"

    @pytest.mark.asyncio
    async def test_sentence_output_aiter(self):
        """Test SentenceOutput async iteration."""
        display_text = DisplayText(text="Test message", name="AI")
        actions = Actions(expressions=["neutral"])
        output = SentenceOutput(
            display_text=display_text,
            tts_text="Test message",
            actions=actions,
        )

        results = []
        async for item in output:
            results.append(item)

        assert len(results) == 1
        result_display, result_tts, result_actions = results[0]
        assert result_display.text == "Test message"
        assert result_tts == "Test message"
        assert result_actions.expressions == ["neutral"]


class TestAudioOutput:
    """Tests for AudioOutput dataclass."""

    def test_audio_output_creation(self):
        """Test AudioOutput creation."""
        display_text = DisplayText(text="Audio message")
        actions = Actions()
        output = AudioOutput(
            audio_path="/path/to/audio.wav",
            display_text=display_text,
            transcript="Audio message",
            actions=actions,
        )
        assert output.audio_path == "/path/to/audio.wav"
        assert output.display_text.text == "Audio message"
        assert output.transcript == "Audio message"

    def test_audio_output_with_actions(self):
        """Test AudioOutput with actions."""
        display_text = DisplayText(text="Excited!")
        actions = Actions(
            expressions=["excited"],
            sounds=["cheer.wav"],
        )
        output = AudioOutput(
            audio_path="/audio.wav",
            display_text=display_text,
            transcript="Excited!",
            actions=actions,
        )
        assert output.actions.expressions == ["excited"]
        assert output.actions.sounds == ["cheer.wav"]

    @pytest.mark.asyncio
    async def test_audio_output_aiter(self):
        """Test AudioOutput async iteration."""
        display_text = DisplayText(text="Test audio", name="Speaker")
        actions = Actions(expressions=["speaking"])
        output = AudioOutput(
            audio_path="/test/audio.mp3",
            display_text=display_text,
            transcript="Test audio transcript",
            actions=actions,
        )

        results = []
        async for item in output:
            results.append(item)

        assert len(results) == 1
        audio_path, result_display, transcript, result_actions = results[0]
        assert audio_path == "/test/audio.mp3"
        assert result_display.text == "Test audio"
        assert transcript == "Test audio transcript"
        assert result_actions.expressions == ["speaking"]

    def test_audio_output_different_display_and_transcript(self):
        """Test AudioOutput with different display text and transcript."""
        display_text = DisplayText(text="[Speaking in Japanese]")
        output = AudioOutput(
            audio_path="/audio.wav",
            display_text=display_text,
            transcript="Konnichiwa",  # Original transcript
            actions=Actions(),
        )
        assert output.display_text.text == "[Speaking in Japanese]"
        assert output.transcript == "Konnichiwa"
