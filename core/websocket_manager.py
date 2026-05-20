"""
WebSocket Manager for real-time attendance recognition system.

Production-grade WebSocket connection manager with:
- Per-connection message queues and heartbeat monitoring
- Rate limiting (sliding window)
- Circuit breaker pattern for fault tolerance
- Reconnection support with session tracking
- Connection metrics and observability
- Graceful shutdown

Attendance rules for classes 5-6-7, shift 2:
  - Present:    12:00 - 13:10  -> status "present",  label "V Khozir"
  - Late:       13:11 - 13:50  -> status "late",     label "Dermonda"
  - Absent:     after 13:50    -> status "absent",   label "Goib"

These rules apply automatically based on the recognition timestamp.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import time as time_module
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shift 2 time boundaries for classes 5-6-7
# ---------------------------------------------------------------------------
SHIFT2_PRESENT_START = time(12, 0)
SHIFT2_PRESENT_END = time(13, 10)
SHIFT2_LATE_END = time(13, 50)


# ---------------------------------------------------------------------------
# WebSocket Protocol (no fastapi dependency required)
# ---------------------------------------------------------------------------
@runtime_checkable
class WebSocketProtocol(Protocol):
    """Protocol defining the WebSocket interface for testability without fastapi."""

    async def accept(self) -> None:
        ...

    async def send_json(self, data: Any) -> None:
        ...

    async def close(self, code: int = 1000) -> None:
        ...

    async def ping(self) -> None:
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class WebSocketConfig:
    """All tunable parameters for the WebSocket manager."""

    max_connections_per_room: int = 100
    ping_interval_seconds: int = 30
    ping_timeout_seconds: int = 10
    message_buffer_size: int = 1000
    rate_limit_messages_per_second: int = 50
    rate_limit_window_seconds: float = 1.0
    reconnect_window_seconds: int = 300
    max_reconnect_attempts: int = 5
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_seconds: int = 60


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class RecognitionEvent:
    """Recognition event produced by the face-recognition pipeline."""

    student_id: str
    name: str
    class_name: str
    confidence: float
    camera_id: int
    timestamp: datetime
    face_box: Dict[str, float] = field(default_factory=dict)
    # face_box expected keys: x, y, w, h


@dataclass
class ConnectionInfo:
    """Per-connection metadata and state."""

    connection_id: str
    session_id: str
    room: str
    websocket: Any  # WebSocketProtocol at runtime
    connected_at: datetime
    last_active: datetime
    client_info: Dict[str, Any] = field(default_factory=dict)
    message_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue())  # type: ignore[type-arg]
    failure_count: int = 0
    is_alive: bool = True
    disconnected_at: Optional[datetime] = None
    reconnect_attempts: int = 0


@dataclass
class ConnectionMetrics:
    """Observability metrics for WebSocket connections."""

    total_connections: int = 0
    active_connections: int = 0
    total_messages_sent: int = 0
    total_messages_failed: int = 0
    total_broadcasts: int = 0
    _broadcast_latencies: List[float] = field(default_factory=list)

    @property
    def average_broadcast_latency_ms(self) -> float:
        if not self._broadcast_latencies:
            return 0.0
        return sum(self._broadcast_latencies) / len(self._broadcast_latencies)

    def record_message_sent(self) -> None:
        """Record a successful message send."""
        self.total_messages_sent += 1

    def record_message_failed(self) -> None:
        """Record a failed message send."""
        self.total_messages_failed += 1

    def record_broadcast(self, latency_ms: float) -> None:
        """Record a broadcast event with its latency."""
        self.total_broadcasts += 1
        self._broadcast_latencies.append(latency_ms)
        # Keep only last 1000 latency records to prevent memory growth
        if len(self._broadcast_latencies) > 1000:
            self._broadcast_latencies = self._broadcast_latencies[-500:]

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as a dictionary."""
        return {
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "total_messages_sent": self.total_messages_sent,
            "total_messages_failed": self.total_messages_failed,
            "total_broadcasts": self.total_broadcasts,
            "average_broadcast_latency_ms": self.average_broadcast_latency_ms,
        }


# ---------------------------------------------------------------------------
# Rate Limiter (sliding window)
# ---------------------------------------------------------------------------
class RateLimiter:
    """Sliding window rate limiter per connection_id."""

    def __init__(self, config: WebSocketConfig) -> None:
        self._config: WebSocketConfig = config
        self._windows: Dict[str, deque] = {}  # type: ignore[type-arg]

    def allow(self, connection_id: str) -> bool:
        """Check if a message from this connection is allowed under rate limits."""
        now: float = time_module.monotonic()
        window_start: float = now - self._config.rate_limit_window_seconds

        if connection_id not in self._windows:
            self._windows[connection_id] = deque()

        window: deque = self._windows[connection_id]  # type: ignore[type-arg]

        # Remove timestamps outside the window
        while window and window[0] < window_start:
            window.popleft()

        if len(window) >= self._config.rate_limit_messages_per_second:
            logger.warning(
                "Rate limit exceeded for connection",
                extra={"connection_id": connection_id},
            )
            return False

        window.append(now)
        return True

    def remove(self, connection_id: str) -> None:
        """Clean up rate limiter state for a disconnected connection."""
        self._windows.pop(connection_id, None)


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------
class CircuitState(enum.Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker pattern for connection fault tolerance."""

    def __init__(self, config: WebSocketConfig) -> None:
        self._config: WebSocketConfig = config
        self._failures: Dict[str, int] = {}
        self._states: Dict[str, CircuitState] = {}
        self._last_failure_time: Dict[str, float] = {}

    def record_failure(self, connection_id: str) -> None:
        """Record a failure for a connection. Transitions to OPEN if threshold reached."""
        self._failures[connection_id] = self._failures.get(connection_id, 0) + 1
        self._last_failure_time[connection_id] = time_module.monotonic()

        if self._failures[connection_id] >= self._config.circuit_breaker_threshold:
            self._states[connection_id] = CircuitState.OPEN
            logger.warning(
                "Circuit breaker opened for connection",
                extra={"connection_id": connection_id},
            )

    def record_success(self, connection_id: str) -> None:
        """Record a success for a connection. Resets to CLOSED state."""
        self._failures[connection_id] = 0
        self._states[connection_id] = CircuitState.CLOSED

    def is_open(self, connection_id: str) -> bool:
        """Check if the circuit breaker is open (blocking sends) for a connection."""
        state: CircuitState = self._states.get(connection_id, CircuitState.CLOSED)

        if state == CircuitState.CLOSED:
            return False

        if state == CircuitState.OPEN:
            # Check if reset timeout has elapsed to transition to half_open
            last_failure: float = self._last_failure_time.get(connection_id, 0.0)
            elapsed: float = time_module.monotonic() - last_failure
            if elapsed >= self._config.circuit_breaker_reset_seconds:
                self._states[connection_id] = CircuitState.HALF_OPEN
                logger.info(
                    "Circuit breaker half-open for connection",
                    extra={"connection_id": connection_id},
                )
                return False
            return True

        # HALF_OPEN: allow one attempt
        return False

    def remove(self, connection_id: str) -> None:
        """Clean up circuit breaker state for a disconnected connection."""
        self._failures.pop(connection_id, None)
        self._states.pop(connection_id, None)
        self._last_failure_time.pop(connection_id, None)

    def remaining_reset_seconds(self, connection_id: str) -> float:
        """Return seconds remaining until the circuit breaker resets for a connection."""
        last_failure: float = self._last_failure_time.get(connection_id, 0.0)
        elapsed: float = time_module.monotonic() - last_failure
        remaining: float = self._config.circuit_breaker_reset_seconds - elapsed
        if remaining < 0.0:
            return 0.0
        return remaining


# ---------------------------------------------------------------------------
# WebSocket Manager
# ---------------------------------------------------------------------------
class WebSocketManager:
    """
    Production-grade WebSocket connection manager.

    Manages connections grouped by rooms with support for:
    - Per-connection message queuing and async delivery
    - Heartbeat/ping-pong monitoring
    - Rate limiting per connection
    - Circuit breaker for fault tolerance
    - Session-based reconnection
    - Graceful shutdown
    - Connection metrics and observability
    """

    def __init__(self, config: Optional[WebSocketConfig] = None) -> None:
        self._config: WebSocketConfig = config or WebSocketConfig()
        self._lock: asyncio.Lock = asyncio.Lock()
        self.connections: Dict[str, Dict[str, ConnectionInfo]] = {}
        self._disconnected_sessions: Dict[str, ConnectionInfo] = {}
        self._metrics: ConnectionMetrics = ConnectionMetrics()
        self._rate_limiter: RateLimiter = RateLimiter(self._config)
        self._circuit_breaker: CircuitBreaker = CircuitBreaker(self._config)
        self._heartbeat_tasks: Dict[str, asyncio.Task] = {}  # type: ignore[type-arg]
        self._consumer_tasks: Dict[str, asyncio.Task] = {}  # type: ignore[type-arg]
        self._shutdown: bool = False

    async def connect(
        self,
        websocket: Any,
        room: str,
        client_info: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Accept and register a WebSocket connection in the specified room.

        Args:
            websocket: WebSocket connection (WebSocketProtocol compatible).
            room: Room identifier to join.
            client_info: Optional metadata about the client.
            session_id: Optional session identifier for reconnection support.

        Returns:
            The generated connection_id (UUID).
        """
        await websocket.accept()

        connection_id: str = str(uuid.uuid4())
        resolved_session_id: str = session_id or str(uuid.uuid4())
        now: datetime = datetime.utcnow()

        info: ConnectionInfo = ConnectionInfo(
            connection_id=connection_id,
            session_id=resolved_session_id,
            room=room,
            websocket=websocket,
            connected_at=now,
            last_active=now,
            client_info=client_info or {},
            message_queue=asyncio.Queue(maxsize=self._config.message_buffer_size),
            failure_count=0,
            is_alive=True,
        )

        async with self._lock:
            if room not in self.connections:
                self.connections[room] = {}

            # Check room capacity
            if len(self.connections[room]) >= self._config.max_connections_per_room:
                logger.warning(
                    "Room at max capacity, rejecting connection",
                    extra={"room": room, "connection_id": connection_id},
                )
                await websocket.close(code=1013)
                return ""

            self.connections[room][connection_id] = info
            self._metrics.total_connections += 1
            self._metrics.active_connections += 1

        # Start background tasks for this connection
        heartbeat_task: asyncio.Task = asyncio.ensure_future(  # type: ignore[type-arg]
            self._heartbeat_loop(connection_id, room)
        )
        self._heartbeat_tasks[connection_id] = heartbeat_task

        consumer_task: asyncio.Task = asyncio.ensure_future(  # type: ignore[type-arg]
            self._message_consumer(connection_id, room)
        )
        self._consumer_tasks[connection_id] = consumer_task

        logger.info(
            "Client connected",
            extra={
                "connection_id": connection_id,
                "session_id": resolved_session_id,
                "room": room,
            },
        )

        return connection_id

    async def disconnect(self, websocket: Any, room: str) -> None:
        """
        Remove a WebSocket connection from the specified room gracefully.

        Args:
            websocket: The WebSocket to disconnect.
            room: The room the WebSocket belongs to.
        """
        connection_id: Optional[str] = None

        async with self._lock:
            if room not in self.connections:
                return

            # Find the connection by websocket reference
            for conn_id, info in self.connections[room].items():
                if info.websocket is websocket:
                    connection_id = conn_id
                    break

            if connection_id is None:
                return

            info = self.connections[room].pop(connection_id)
            info.is_alive = False
            info.disconnected_at = datetime.utcnow()
            self._metrics.active_connections -= 1

            # Store for potential reconnection
            self._disconnected_sessions[info.session_id] = info

            # Sweep stale disconnected sessions to prevent unbounded growth
            self._sweep_disconnected_sessions()

            if not self.connections[room]:
                del self.connections[room]

        # Cancel background tasks
        self._cancel_tasks_for_connection(connection_id)

        # Clean up rate limiter and circuit breaker state
        self._rate_limiter.remove(connection_id)
        self._circuit_breaker.remove(connection_id)

        logger.info(
            "Client disconnected",
            extra={
                "connection_id": connection_id,
                "session_id": info.session_id,
                "room": room,
            },
        )

    async def reconnect(
        self,
        websocket: Any,
        session_id: str,
        room: str,
    ) -> Optional[str]:
        """
        Attempt to reconnect a previously disconnected session.

        Args:
            websocket: The new WebSocket connection.
            session_id: The session_id from a previous connection.
            room: The room to rejoin.

        Returns:
            The new connection_id if reconnection succeeded, None otherwise.
        """
        if session_id not in self._disconnected_sessions:
            logger.warning(
                "Reconnect failed: session not found",
                extra={"session_id": session_id, "room": room},
            )
            return None

        old_info: ConnectionInfo = self._disconnected_sessions[session_id]

        # Check if within reconnect window (measured from disconnect time)
        disconnect_time: Optional[datetime] = old_info.disconnected_at
        if disconnect_time is None:
            # Fallback: treat connected_at as disconnect time
            disconnect_time = old_info.connected_at
        elapsed: float = (datetime.utcnow() - disconnect_time).total_seconds()
        if elapsed > self._config.reconnect_window_seconds:
            logger.warning(
                "Reconnect failed: window expired",
                extra={"session_id": session_id, "room": room},
            )
            del self._disconnected_sessions[session_id]
            return None

        # Check max reconnect attempts
        if old_info.reconnect_attempts >= self._config.max_reconnect_attempts:
            logger.warning(
                "Reconnect failed: max attempts exceeded",
                extra={
                    "session_id": session_id,
                    "room": room,
                    "attempts": old_info.reconnect_attempts,
                },
            )
            del self._disconnected_sessions[session_id]
            return None

        # Accept and re-register
        await websocket.accept()

        # Increment reconnect attempt counter
        old_info.reconnect_attempts += 1

        connection_id: str = str(uuid.uuid4())
        now: datetime = datetime.utcnow()

        info: ConnectionInfo = ConnectionInfo(
            connection_id=connection_id,
            session_id=session_id,
            room=room,
            websocket=websocket,
            connected_at=now,
            last_active=now,
            client_info=old_info.client_info,
            message_queue=asyncio.Queue(maxsize=self._config.message_buffer_size),
            failure_count=0,
            is_alive=True,
            disconnected_at=None,
            reconnect_attempts=old_info.reconnect_attempts,
        )

        async with self._lock:
            if room not in self.connections:
                self.connections[room] = {}

            self.connections[room][connection_id] = info
            self._metrics.total_connections += 1
            self._metrics.active_connections += 1

        # Remove from disconnected sessions
        del self._disconnected_sessions[session_id]

        # Start background tasks
        heartbeat_task: asyncio.Task = asyncio.ensure_future(  # type: ignore[type-arg]
            self._heartbeat_loop(connection_id, room)
        )
        self._heartbeat_tasks[connection_id] = heartbeat_task

        consumer_task: asyncio.Task = asyncio.ensure_future(  # type: ignore[type-arg]
            self._message_consumer(connection_id, room)
        )
        self._consumer_tasks[connection_id] = consumer_task

        logger.info(
            "Client reconnected",
            extra={
                "connection_id": connection_id,
                "session_id": session_id,
                "room": room,
            },
        )

        return connection_id

    async def _heartbeat_loop(self, connection_id: str, room: str) -> None:
        """
        Periodically send WebSocket ping and await pong with timeout.

        If pong is not received within the timeout, the connection is marked
        as stale and disconnected.
        """
        try:
            while not self._shutdown:
                await asyncio.sleep(self._config.ping_interval_seconds)

                info: Optional[ConnectionInfo] = await self._get_connection_info(
                    connection_id, room
                )
                if info is None or not info.is_alive:
                    break

                try:
                    await asyncio.wait_for(
                        info.websocket.ping(),
                        timeout=self._config.ping_timeout_seconds,
                    )
                    info.last_active = datetime.utcnow()
                except (asyncio.TimeoutError, Exception) as exc:
                    logger.warning(
                        "Heartbeat failed, disconnecting",
                        extra={
                            "connection_id": connection_id,
                            "room": room,
                            "error": str(exc),
                        },
                    )
                    await self.disconnect(info.websocket, room)
                    break
        except asyncio.CancelledError:
            pass

    async def _message_consumer(self, connection_id: str, room: str) -> None:
        """
        Read messages from the connection's queue and send them to the WebSocket.

        Integrates with the circuit breaker to handle persistent failures.
        """
        try:
            while not self._shutdown:
                info: Optional[ConnectionInfo] = await self._get_connection_info(
                    connection_id, room
                )
                if info is None or not info.is_alive:
                    break

                try:
                    payload: Dict[str, Any] = await asyncio.wait_for(
                        info.message_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Check circuit breaker
                if self._circuit_breaker.is_open(connection_id):
                    logger.warning(
                        "Circuit open, buffering message",
                        extra={"connection_id": connection_id, "room": room},
                    )
                    # Re-queue the message if possible
                    if not info.message_queue.full():
                        await info.message_queue.put(payload)
                    else:
                        logger.warning(
                            "Queue full while circuit open, dropping message",
                            extra={"connection_id": connection_id, "room": room},
                        )
                        self._metrics.record_message_failed()
                    # Sleep for remaining circuit reset time instead of fixed 1s
                    remaining_reset: float = (
                        self._circuit_breaker.remaining_reset_seconds(connection_id)
                    )
                    await asyncio.sleep(remaining_reset)
                    continue

                try:
                    await info.websocket.send_json(payload)
                    self._circuit_breaker.record_success(connection_id)
                    self._metrics.record_message_sent()
                    info.last_active = datetime.utcnow()
                except Exception as exc:
                    logger.warning(
                        "Failed to send message to client",
                        extra={
                            "connection_id": connection_id,
                            "room": room,
                            "error": str(exc),
                        },
                    )
                    self._circuit_breaker.record_failure(connection_id)
                    self._metrics.record_message_failed()
                    info.failure_count += 1
        except asyncio.CancelledError:
            pass

    async def _get_connection_info(
        self, connection_id: str, room: str
    ) -> Optional[ConnectionInfo]:
        """Get connection info with lock for safe access from async tasks."""
        async with self._lock:
            room_connections: Optional[Dict[str, ConnectionInfo]] = (
                self.connections.get(room)
            )
            if room_connections is None:
                return None
            return room_connections.get(connection_id)

    def _cancel_tasks_for_connection(self, connection_id: str) -> None:
        """Cancel heartbeat and consumer tasks for a connection."""
        heartbeat_task: Optional[asyncio.Task] = self._heartbeat_tasks.pop(  # type: ignore[type-arg]
            connection_id, None
        )
        if heartbeat_task is not None and not heartbeat_task.done():
            heartbeat_task.cancel()

        consumer_task: Optional[asyncio.Task] = self._consumer_tasks.pop(  # type: ignore[type-arg]
            connection_id, None
        )
        if consumer_task is not None and not consumer_task.done():
            consumer_task.cancel()

    def _sweep_disconnected_sessions(self) -> None:
        """Remove stale entries from _disconnected_sessions that have exceeded the reconnect window."""
        now: datetime = datetime.utcnow()
        stale_keys: List[str] = []
        for session_id, info in self._disconnected_sessions.items():
            disconnect_time: Optional[datetime] = info.disconnected_at
            if disconnect_time is None:
                # Fallback: treat connected_at as disconnect time
                disconnect_time = info.connected_at
            elapsed: float = (now - disconnect_time).total_seconds()
            if elapsed > self._config.reconnect_window_seconds:
                stale_keys.append(session_id)
        for key in stale_keys:
            del self._disconnected_sessions[key]

    async def _send_to_connection(
        self, connection_id: str, room: str, payload: Dict[str, Any]
    ) -> None:
        """
        Enqueue a message payload to a specific connection's queue.

        Checks rate limiter before enqueuing. If rate limit is exceeded,
        the message is dropped.
        """
        if not self._rate_limiter.allow(connection_id):
            logger.warning(
                "Rate limit exceeded, dropping message",
                extra={"connection_id": connection_id, "room": room},
            )
            return

        info: Optional[ConnectionInfo] = await self._get_connection_info(
            connection_id, room
        )
        if info is None or not info.is_alive:
            return

        if info.message_queue.full():
            # Drop oldest message to make room
            try:
                info.message_queue.get_nowait()
                logger.warning(
                    "Queue full, dropped oldest message",
                    extra={"connection_id": connection_id, "room": room},
                )
            except asyncio.QueueEmpty:
                pass

        try:
            info.message_queue.put_nowait(payload)
        except asyncio.QueueFull:
            logger.error(
                "Failed to enqueue message",
                extra={"connection_id": connection_id, "room": room},
            )
            self._metrics.record_message_failed()

    async def _send_to_room(self, room: str, payload: Dict[str, Any]) -> None:
        """
        Send a JSON payload to all connected clients in the specified room.

        Uses the lock when reading connections and tracks broadcast latency.
        """
        start_time: float = time_module.monotonic()

        async with self._lock:
            room_connections: Optional[Dict[str, ConnectionInfo]] = (
                self.connections.get(room)
            )
            if room_connections is None:
                return
            # Copy connection ids to avoid mutation during iteration
            connection_ids: List[str] = list(room_connections.keys())

        for conn_id in connection_ids:
            await self._send_to_connection(conn_id, room, payload)

        elapsed_ms: float = (time_module.monotonic() - start_time) * 1000.0
        self._metrics.record_broadcast(elapsed_ms)

    async def broadcast(self, room: str, payload: Dict[str, Any]) -> None:
        """
        Public method to broadcast an arbitrary payload to all clients in a room.

        Args:
            room: Target room identifier.
            payload: JSON-serializable payload dict.
        """
        await self._send_to_room(room, payload)

    async def broadcast_recognition(self, event: RecognitionEvent) -> None:
        """
        Broadcast a recognition event to all clients in the 'attendance' room.

        Automatically determines status and label for shift 2 classes 5-6-7
        based on the event timestamp.
        """
        status: str
        status_label: str
        status, status_label = self.determine_shift2_status(event.timestamp)

        payload: Dict[str, Any] = {
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
        payload: Dict[str, Any] = {
            "type": "camera_status",
            "camera_id": cam_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self._send_to_room("attendance", payload)

    async def broadcast_session_update(
        self, session_id: int, stats: Dict[str, int]
    ) -> None:
        """
        Broadcast session statistics update to all clients in the 'attendance' room.

        Args:
            session_id: The active session identifier.
            stats: Dictionary with keys 'present', 'late', 'absent' and their counts.
        """
        payload: Dict[str, Any] = {
            "type": "session_stats",
            "session_id": session_id,
            "present": stats.get("present", 0),
            "late": stats.get("late", 0),
            "absent": stats.get("absent", 0),
            "timestamp": datetime.utcnow().isoformat(),
        }

        await self._send_to_room("attendance", payload)

    def determine_shift2_status(self, timestamp: datetime) -> Tuple[str, str]:
        """
        Determine attendance status for classes 5-6-7 shift 2
        based on the recognition timestamp.

        Time windows:
          12:00 - 13:10 -> present
          13:11 - 13:50 -> late
          after 13:50   -> absent

        Returns:
            Tuple of (status, status_label)
        """
        current_time: time = timestamp.time()

        if SHIFT2_PRESENT_START <= current_time <= SHIFT2_PRESENT_END:
            return "present", "\u2713 \u04b2\u043e\u0437\u0438\u0440"
        elif current_time <= SHIFT2_LATE_END:
            return "late", "\u23f0 \u0414\u0435\u0440\u043c\u043e\u043d\u0434\u0430"
        else:
            return "absent", "\u2717 \u0413\u043e\u0438\u0431"

    async def shutdown(self) -> None:
        """
        Gracefully shut down the manager.

        Closes all connections with code 1001 (going away), cancels all
        background tasks, and clears all data structures.
        """
        self._shutdown = True
        logger.info("WebSocket manager shutting down")

        async with self._lock:
            all_connections: List[Tuple[str, str, ConnectionInfo]] = []
            for room, room_conns in self.connections.items():
                for conn_id, info in room_conns.items():
                    all_connections.append((room, conn_id, info))

        # Close all connections
        for room, conn_id, info in all_connections:
            try:
                await info.websocket.close(code=1001)
            except Exception as exc:
                logger.warning(
                    "Error closing connection during shutdown",
                    extra={
                        "connection_id": conn_id,
                        "room": room,
                        "error": str(exc),
                    },
                )

            self._cancel_tasks_for_connection(conn_id)

        # Clear all data structures
        async with self._lock:
            self.connections.clear()
            self._disconnected_sessions.clear()
            self._metrics.active_connections = 0

        logger.info("WebSocket manager shutdown complete")

    def get_metrics(self) -> Dict[str, Any]:
        """Return connection metrics as a dictionary."""
        return self._metrics.to_dict()

    def get_room_info(self, room: str) -> Dict[str, Any]:
        """
        Return information about a specific room.

        Args:
            room: Room identifier.

        Returns:
            Dictionary with connection count and connection metadata.
        """
        room_connections: Optional[Dict[str, ConnectionInfo]] = self.connections.get(
            room
        )
        if room_connections is None:
            return {"room": room, "connection_count": 0, "connections": []}

        connections_info: List[Dict[str, Any]] = []
        for conn_id, info in room_connections.items():
            connections_info.append(
                {
                    "connection_id": info.connection_id,
                    "session_id": info.session_id,
                    "connected_at": info.connected_at.isoformat(),
                    "last_active": info.last_active.isoformat(),
                    "client_info": info.client_info,
                    "is_alive": info.is_alive,
                    "failure_count": info.failure_count,
                }
            )

        return {
            "room": room,
            "connection_count": len(room_connections),
            "connections": connections_info,
        }
