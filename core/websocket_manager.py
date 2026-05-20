"""
WebSocket Manager for real-time attendance recognition system.

Attendance rules for classes 5-6-7, shift 2 (бахши ду):
  - Present (Хозир):    12:00 - 13:10  -> status "present",  label "✓ Хозир"
  - Late (Дермонда):    13:11 - 13:50  -> status "late",     label "⏰ Дермонда"
  - Absent (Гоиб):     after 13:50     -> status "absent",   label "✗ Гоиб"

These rules apply automatically based on the recognition timestamp.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


# Shift 2 time boundaries for classes 5-6-7
SHIFT2_PRESENT_START = time(12, 0)
SHIFT2_PRESENT_END = time(13, 10)
SHIFT2_LATE_END = time(13, 50)


@dataclass
class RecognitionEvent:
    """Recognition event produced by the face-recognition pipeline."""

    student_id: str
    name: str
    class_name: str
    confidence: float
    camera_id: int
    timestamp: datetime
    face_box: dict[str, float] = field(default_factory=dict)
    # face_box expected keys: x, y, w, h


class WebSocketManager:
    """
    Manages WebSocket connections grouped by rooms and broadcasts
    real-time attendance events to connected clients.
    """

    def __init__(self) -> None:
        self.connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room: str) -> None:
        """Accept and register a WebSocket connection in the specified room."""
        await websocket.accept()
        if room not in self.connections:
            self.connections[room] = []
        self.connections[room].append(websocket)
        logger.info("Client connected to room '%s'. Total in room: %d", room, len(self.connections[room]))

    async def disconnect(self, websocket: WebSocket, room: str) -> None:
        """Remove a WebSocket connection from the specified room gracefully."""
        if room in self.connections:
            try:
                self.connections[room].remove(websocket)
            except ValueError:
                pass
            if not self.connections[room]:
                del self.connections[room]
        logger.info("Client disconnected from room '%s'.", room)

    def determine_shift2_status(self, timestamp: datetime) -> tuple[str, str]:
        """
        Determine attendance status for classes 5-6-7 shift 2 (бахши ду)
        based on the recognition timestamp.

        Time windows:
          12:00 - 13:10 -> present  (✓ Хозир)
          13:11 - 13:50 -> late     (⏰ Дермонда)
          after 13:50   -> absent   (✗ Гоиб)

        Returns:
            Tuple of (status, status_label)
        """
        current_time = timestamp.time()

        if SHIFT2_PRESENT_START <= current_time <= SHIFT2_PRESENT_END:
            return "present", "\u2713 \u04b2\u043e\u0437\u0438\u0440"
        elif current_time <= SHIFT2_LATE_END:
            return "late", "\u23f0 \u0414\u0435\u0440\u043c\u043e\u043d\u0434\u0430"
        else:
            return "absent", "\u2717 \u0413\u043e\u0438\u0431"

    async def broadcast_recognition(self, event: RecognitionEvent) -> None:
        """
        Broadcast a recognition event to all clients in the 'attendance' room.

        Automatically determines status and label for shift 2 classes 5-6-7
        based on the event timestamp.
        """
        status, status_label = self.determine_shift2_status(event.timestamp)

        payload: dict[str, Any] = {
            "type": "recognition",
            "student_id": event.student_id,
            "name": event.name,
            "class_name": event.class_name,
            "status": status,
            "status_label": status_label,
            "confidence": event.confidence,
            "camera_id": event.camera_id,
            "timestamp": event.timestamp.isoformat(),
            "face_box": event.face_box,
        }

        await self._send_to_room("attendance", payload)

    async def broadcast_camera_status(self, cam_id: int, status: str) -> None:
        """Broadcast camera status change to all clients in the 'attendance' room."""
        payload: dict[str, Any] = {
            "type": "camera_status",
            "camera_id": cam_id,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }

        await self._send_to_room("attendance", payload)

    async def broadcast_session_update(self, session_id: int, stats: dict[str, int]) -> None:
        """
        Broadcast session statistics update to all clients in the 'attendance' room.

        Args:
            session_id: The active session identifier.
            stats: Dictionary with keys 'present', 'late', 'absent' and their counts.
        """
        payload: dict[str, Any] = {
            "type": "session_stats",
            "session_id": session_id,
            "present": stats.get("present", 0),
            "late": stats.get("late", 0),
            "absent": stats.get("absent", 0),
            "timestamp": datetime.now().isoformat(),
        }

        await self._send_to_room("attendance", payload)

    async def _send_to_room(self, room: str, payload: dict[str, Any]) -> None:
        """Send a JSON payload to all connected clients in the specified room."""
        if room not in self.connections:
            return

        disconnected: list[WebSocket] = []

        for websocket in self.connections[room]:
            try:
                await websocket.send_json(payload)
            except Exception as e:
                logger.warning("Failed to send to client in room '%s': %s", room, e)
                disconnected.append(websocket)

        # Clean up broken connections
        for ws in disconnected:
            await self.disconnect(ws, room)
