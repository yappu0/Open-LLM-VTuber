"""
Tests for input types (input_types.py).
"""

import pytest

from src.open_llm_vtuber.agent.input_types import (
    BatchInput,
    FileData,
    ImageData,
    ImageSource,
    TextData,
    TextSource,
)


class TestTextSource:
    """Tests for TextSource enum."""

    def test_text_source_values(self):
        """Test TextSource enum values."""
        assert TextSource.INPUT.value == "input"
        assert TextSource.CLIPBOARD.value == "clipboard"

    def test_text_source_membership(self):
        """Test TextSource enum membership."""
        assert TextSource.INPUT in TextSource
        assert TextSource.CLIPBOARD in TextSource


class TestImageSource:
    """Tests for ImageSource enum."""

    def test_image_source_values(self):
        """Test ImageSource enum values."""
        assert ImageSource.CAMERA.value == "camera"
        assert ImageSource.SCREEN.value == "screen"
        assert ImageSource.CLIPBOARD.value == "clipboard"
        assert ImageSource.UPLOAD.value == "upload"


class TestTextData:
    """Tests for TextData dataclass."""

    def test_text_data_creation(self):
        """Test basic TextData creation."""
        text_data = TextData(
            source=TextSource.INPUT,
            content="Hello, world!",
        )
        assert text_data.source == TextSource.INPUT
        assert text_data.content == "Hello, world!"
        assert text_data.from_name is None

    def test_text_data_with_from_name(self):
        """Test TextData creation with from_name."""
        text_data = TextData(
            source=TextSource.CLIPBOARD,
            content="Copied text",
            from_name="User",
        )
        assert text_data.source == TextSource.CLIPBOARD
        assert text_data.content == "Copied text"
        assert text_data.from_name == "User"

    def test_text_data_empty_content(self):
        """Test TextData with empty content."""
        text_data = TextData(
            source=TextSource.INPUT,
            content="",
        )
        assert text_data.content == ""


class TestImageData:
    """Tests for ImageData dataclass."""

    def test_image_data_creation(self):
        """Test basic ImageData creation."""
        image_data = ImageData(
            source=ImageSource.CAMERA,
            data="base64_encoded_data",
            mime_type="image/jpeg",
        )
        assert image_data.source == ImageSource.CAMERA
        assert image_data.data == "base64_encoded_data"
        assert image_data.mime_type == "image/jpeg"

    def test_image_data_from_screen(self):
        """Test ImageData from screen source."""
        image_data = ImageData(
            source=ImageSource.SCREEN,
            data="screenshot_data",
            mime_type="image/png",
        )
        assert image_data.source == ImageSource.SCREEN
        assert image_data.mime_type == "image/png"

    def test_image_data_from_upload(self):
        """Test ImageData from upload source."""
        image_data = ImageData(
            source=ImageSource.UPLOAD,
            data="uploaded_image_base64",
            mime_type="image/gif",
        )
        assert image_data.source == ImageSource.UPLOAD


class TestFileData:
    """Tests for FileData dataclass."""

    def test_file_data_creation(self):
        """Test basic FileData creation."""
        file_data = FileData(
            name="document.pdf",
            data="base64_pdf_data",
            mime_type="application/pdf",
        )
        assert file_data.name == "document.pdf"
        assert file_data.data == "base64_pdf_data"
        assert file_data.mime_type == "application/pdf"

    def test_file_data_text_file(self):
        """Test FileData for text file."""
        file_data = FileData(
            name="readme.txt",
            data="dGV4dCBjb250ZW50",  # "text content" in base64
            mime_type="text/plain",
        )
        assert file_data.name == "readme.txt"
        assert file_data.mime_type == "text/plain"


class TestBatchInput:
    """Tests for BatchInput dataclass."""

    def test_batch_input_minimal(self):
        """Test BatchInput with minimal required fields."""
        batch_input = BatchInput(
            texts=[TextData(source=TextSource.INPUT, content="Hello")],
        )
        assert len(batch_input.texts) == 1
        assert batch_input.texts[0].content == "Hello"
        assert batch_input.images is None
        assert batch_input.files is None
        assert batch_input.metadata is None

    def test_batch_input_with_images(self):
        """Test BatchInput with images."""
        batch_input = BatchInput(
            texts=[TextData(source=TextSource.INPUT, content="Look at this")],
            images=[
                ImageData(
                    source=ImageSource.CAMERA,
                    data="image_data",
                    mime_type="image/jpeg",
                )
            ],
        )
        assert len(batch_input.images) == 1
        assert batch_input.images[0].source == ImageSource.CAMERA

    def test_batch_input_with_files(self):
        """Test BatchInput with files."""
        batch_input = BatchInput(
            texts=[TextData(source=TextSource.INPUT, content="Check this file")],
            files=[
                FileData(
                    name="doc.pdf",
                    data="pdf_data",
                    mime_type="application/pdf",
                )
            ],
        )
        assert len(batch_input.files) == 1
        assert batch_input.files[0].name == "doc.pdf"

    def test_batch_input_with_metadata(self):
        """Test BatchInput with metadata."""
        batch_input = BatchInput(
            texts=[TextData(source=TextSource.INPUT, content="Proactive message")],
            metadata={
                "proactive_speak": True,
                "skip_memory": True,
                "skip_history": False,
            },
        )
        assert batch_input.metadata["proactive_speak"] is True
        assert batch_input.metadata["skip_memory"] is True
        assert batch_input.metadata["skip_history"] is False

    def test_batch_input_multiple_texts(self):
        """Test BatchInput with multiple text sources."""
        batch_input = BatchInput(
            texts=[
                TextData(source=TextSource.INPUT, content="Main input"),
                TextData(
                    source=TextSource.CLIPBOARD,
                    content="Clipboard content",
                    from_name="User",
                ),
            ],
        )
        assert len(batch_input.texts) == 2
        assert batch_input.texts[0].source == TextSource.INPUT
        assert batch_input.texts[1].source == TextSource.CLIPBOARD

    def test_batch_input_full(self):
        """Test BatchInput with all fields populated."""
        batch_input = BatchInput(
            texts=[
                TextData(source=TextSource.INPUT, content="Full test"),
            ],
            images=[
                ImageData(
                    source=ImageSource.UPLOAD,
                    data="img1",
                    mime_type="image/png",
                ),
                ImageData(
                    source=ImageSource.SCREEN,
                    data="img2",
                    mime_type="image/jpeg",
                ),
            ],
            files=[
                FileData(name="file1.txt", data="data1", mime_type="text/plain"),
            ],
            metadata={"custom_key": "custom_value"},
        )
        assert len(batch_input.texts) == 1
        assert len(batch_input.images) == 2
        assert len(batch_input.files) == 1
        assert batch_input.metadata["custom_key"] == "custom_value"
