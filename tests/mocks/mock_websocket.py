"""Mock WebSocket class for testing."""

import json
from typing import List, Any, Optional


class MockWebSocket:
    """Mock WebSocket for testing message sending and receiving."""

    def __init__(self):
        """Initialize the mock WebSocket."""
        self.sent_messages: List[str] = []
        self.sent_bytes: List[bytes] = []
        self.receive_queue: List[Any] = []
        self.closed = False
        self.accepted = False

    async def accept(self) -> None:
        """Mock accepting the WebSocket connection."""
        self.accepted = True

    async def send_text(self, message: str) -> None:
        """
        Mock sending a text message.

        Args:
            message: The text message to send.
        """
        self.sent_messages.append(message)

    async def send_bytes(self, data: bytes) -> None:
        """
        Mock sending binary data.

        Args:
            data: The binary data to send.
        """
        self.sent_bytes.append(data)

    async def send_json(self, data: Any) -> None:
        """
        Mock sending JSON data.

        Args:
            data: The data to serialize and send.
        """
        self.sent_messages.append(json.dumps(data))

    async def receive_text(self) -> str:
        """
        Mock receiving a text message.

        Returns:
            The next message in the receive queue.

        Raises:
            IndexError: If there are no messages in the queue.
        """
        if not self.receive_queue:
            raise IndexError("No messages in receive queue")
        return self.receive_queue.pop(0)

    async def receive_json(self) -> Any:
        """
        Mock receiving JSON data.

        Returns:
            The parsed JSON from the next message in the queue.

        Raises:
            IndexError: If there are no messages in the queue.
        """
        text = await self.receive_text()
        return json.loads(text)

    async def close(self, code: int = 1000, reason: Optional[str] = None) -> None:
        """
        Mock closing the WebSocket connection.

        Args:
            code: The close code.
            reason: The close reason.
        """
        self.closed = True

    def queue_message(self, message: Any) -> None:
        """
        Add a message to the receive queue.

        Args:
            message: The message to add (will be JSON serialized if not a string).
        """
        if isinstance(message, str):
            self.receive_queue.append(message)
        else:
            self.receive_queue.append(json.dumps(message))

    def get_sent_json_messages(self) -> List[Any]:
        """
        Get all sent messages parsed as JSON.

        Returns:
            List of parsed JSON messages.
        """
        result = []
        for msg in self.sent_messages:
            try:
                result.append(json.loads(msg))
            except json.JSONDecodeError:
                result.append(msg)
        return result

    def clear(self) -> None:
        """Clear all sent messages and the receive queue."""
        self.sent_messages.clear()
        self.sent_bytes.clear()
        self.receive_queue.clear()
