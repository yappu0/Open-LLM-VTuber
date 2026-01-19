import abc
import os
import re
import asyncio

from loguru import logger


class TTSInterface(metaclass=abc.ABCMeta):
    async def async_generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """
        Asynchronously generate speech audio file using TTS.

        By default, this runs the synchronous generate_audio in a coroutine.
        Subclasses can override this method to provide true async implementation.

        text: str
            the text to speak
        file_name_no_ext (optional and deprecated): str
            name of the file without file extension

        Returns:
        str: the path to the generated audio file

        """
        return await asyncio.to_thread(self.generate_audio, text, file_name_no_ext)

    @abc.abstractmethod
    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """
        Generate speech audio file using TTS.
        text: str
            the text to speak
        file_name_no_ext (optional and deprecated): str
            name of the file without file extension

        Returns:
        str: the path to the generated audio file

        """
        raise NotImplementedError

    def remove_file(self, filepath: str, verbose: bool = True) -> None:
        """
        Remove a file from the file system.

        This is a separate method instead of a part of the `play_audio_file_local()` because `play_audio_file_local()` is not the only way to play audio files. This method will be used to remove the audio file after it has been played.

        Parameters:
            filepath (str): The path to the file to remove.
            verbose (bool): If True, print messages to the console.
        """
        if not os.path.exists(filepath):
            logger.warning(f"File {filepath} does not exist")
            return
        try:
            logger.debug(f"Removing file {filepath}") if verbose else None
            os.remove(filepath)
        except Exception as e:
            logger.error(f"Failed to remove file {filepath}: {e}")

    # Allowed audio file extensions for cache files
    ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a", "aac", "opus"}

    def generate_cache_file_name(self, file_name_no_ext=None, file_extension="wav"):
        """
        Generate a cross-platform cache file name with path traversal protection.

        file_name_no_ext: str
            name of the file without extension
        file_extension: str
            file extension (must be one of: wav, mp3, ogg, flac, m4a, aac, opus)

        Returns:
        str: the path to the generated cache file
        """
        cache_dir = "cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        if file_name_no_ext is None:
            file_name_no_ext = "temp"

        # Security: Sanitize file_name_no_ext to prevent path traversal
        # Remove any path separators and use only the basename
        safe_name = os.path.basename(
            file_name_no_ext.replace("/", "_").replace("\\", "_")
        )
        if not safe_name:
            safe_name = "temp"

        # Additional validation: only allow alphanumeric, underscore, hyphen
        if not re.match(r"^[\w\-]+$", safe_name):
            logger.warning(
                f"Unsafe filename detected, using 'temp': {file_name_no_ext}"
            )
            safe_name = "temp"

        # Security: Validate file extension against allowlist to prevent path traversal
        safe_extension = file_extension.lower().lstrip(".")
        if safe_extension not in self.ALLOWED_EXTENSIONS:
            logger.warning(f"Invalid file extension '{file_extension}', using 'wav'")
            safe_extension = "wav"

        file_name = f"{safe_name}.{safe_extension}"
        return os.path.join(cache_dir, file_name)
