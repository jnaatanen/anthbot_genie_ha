"""Data coordinator for Anthbot Genie."""

from __future__ import annotations

import base64
from datetime import timedelta
import logging
import struct
import time
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .api import (
    AnthbotBoundDevice,
    AnthbotCloudApiClient,
    AnthbotGenieApiError,
    AnthbotShadowApiClient,
)
from .const import DOMAIN

_CURPATH_MAGIC = b"\x16\x01\x03\x05"
_CURPATH_HEADER_LEN = 22
_CURPATH_RECORD_LEN = 5
_CURPATH_SCALE = 10  # curpath is in centimetres; pose/zone vertexs are in millimetres
_COVERAGE_MAX_POINTS = 5000
# While the mower is idle/docked there is no live telemetry worth polling for, so
# the coordinator falls back to this slow cadence to avoid hammering the AWS IoT
# shadow endpoint (which returns HTTP 429 TOO_MANY_REQUESTS under frequent polls).
# While actively mowing it uses the configured (fast) interval + keep-alive.
_IDLE_SCAN_INTERVAL = timedelta(minutes=10)
# After a command issued from Home Assistant (e.g. start mowing) the coordinator
# keeps polling at the fast cadence for this long, so a command is reflected
# quickly and the transition into a mowing state is caught even if the mower was
# in the slow idle cadence and takes a few seconds to start.
_COMMAND_ACTIVE_GRACE = 120.0  # seconds
# If the shadow endpoint still rate-limits us (HTTP 429), back off exponentially
# from this start, doubling each consecutive failure, capped at the max.
_RATE_LIMIT_BACKOFF_START = timedelta(minutes=1)
_RATE_LIMIT_BACKOFF_MAX = timedelta(minutes=30)
_MOWING_STATES = {
    "globalmowing", "zonemowing", "pointmowing",
    "bordermowing", "regionmowing", "nestmowing",
}
_DOCK_RESET_STATES = {
    "charge", "charging", "charge_start", "backtodock",
    "idle", "sleep", "shutdown",
}

# Persistent yard map: a permanent, cross-session point cloud (deduped on a grid)
# that survives restarts and only resets when the device's map identity changes.
_YARD_GRID_MM = 250  # dedup grid cell size in millimetres
_YARD_MAX_CELLS = 10000  # ~625 m² at the 250 mm grid; covers large yards
_YARD_SAVE_DELAY = 30  # seconds to batch Store writes
_YARD_STORE_VERSION = 1
_YARD_REMAP_CONFIRM_POLLS = 3  # a new map_id must persist this many polls before reset


def _decode_curpath_mm(blob: Any) -> list[list[int]]:
    """Decode the base64 ``curpath`` window into ``[x, y]`` points in millimetres.

    22-byte header (magic ``16 01 03 05``, uint32 LE point count at offset 4),
    then that many 5-byte records of ``int16 x, int16 y`` (little-endian) plus a
    1-byte flag. The raw values are in CENTIMETRES, so they are scaled to
    millimetres (the same frame as the zone ``vertexs`` and ``pose``).
    """
    if not isinstance(blob, str) or not blob:
        return []
    try:
        raw = base64.b64decode(blob)
    except (ValueError, TypeError):
        return []
    if len(raw) < _CURPATH_HEADER_LEN + _CURPATH_RECORD_LEN or raw[:4] != _CURPATH_MAGIC:
        return []
    count = struct.unpack_from("<I", raw, 4)[0]
    body = raw[_CURPATH_HEADER_LEN:]
    usable = min(count, len(body) // _CURPATH_RECORD_LEN)
    points: list[list[int]] = []
    for i in range(usable):
        x, y = struct.unpack_from("<hh", body, i * _CURPATH_RECORD_LEN)
        points.append([x * _CURPATH_SCALE, y * _CURPATH_SCALE])
    return points


def _raw_robot_status(data: dict[str, Any]) -> str | None:
    """Return the raw robot status string (Genie 600 ``robot_sta`` / M5-M9 ``mode``)."""
    for key in ("robot_sta", "mode"):
        value = data.get(key)
        if isinstance(value, dict):
            raw = value.get("value")
            if isinstance(raw, str):
                return raw.lower()
    return None


def _device_map_id(data: dict[str, Any]) -> str | None:
    """Return the device's real map id, or None if it does not report one.

    ONLY ``multi_maps.map_list[0].map_id`` is trusted as a map identity. The
    timestamp fields (``map_time`` / ``map_tar_time`` / ``area_time``) are NOT
    used: ``area_time`` provably changes on a plain zone edit (no re-map), and
    mixing sources made the persistent yard map reset spuriously. When the
    device reports no real ``map_id``, this returns None and the yard map is
    NEVER auto-reset (it only grows; use the manual reset button to clear it).
    """
    multi = data.get("multi_maps")
    if isinstance(multi, dict):
        map_list = multi.get("map_list")
        if isinstance(map_list, list) and map_list and isinstance(map_list[0], dict):
            map_id = map_list[0].get("map_id")
            if isinstance(map_id, str) and map_id:
                return map_id
    return None


def _simplify_collinear(points: list[list[int]]) -> list[list[int]]:
    """Drop vertices that lie on a straight segment between their neighbours."""
    n = len(points)
    if n < 3:
        return points
    out: list[list[int]] = []
    for i in range(n):
        ax, ay = points[i - 1]
        bx, by = points[i]
        cx, cy = points[(i + 1) % n]
        # Keep b only if a-b-c is not collinear (cross product != 0).
        if (bx - ax) * (cy - by) != (by - ay) * (cx - bx):
            out.append([bx, by])
    return out


_YARD_BOUNDARY_CLOSE_CELLS = 2  # light closing (~500 mm) to bridge within-region poll gaps
_YARD_MIN_COMPONENT_CELLS = 12  # ignore coverage blobs smaller than this (noise)


def _connected_components(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Split occupied cells into 4-connected components."""
    remaining = set(cells)
    components: list[set[tuple[int, int]]] = []
    while remaining:
        start = next(iter(remaining))
        comp = {start}
        stack = [start]
        remaining.discard(start)
        while stack:
            cx, cy = stack.pop()
            for nb in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if nb in remaining:
                    remaining.discard(nb)
                    comp.add(nb)
                    stack.append(nb)
        components.append(comp)
    return components


def _dilate_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """One-cell morphological dilation (add the 4-neighbours of every cell)."""
    out = set(cells)
    for cx, cy in cells:
        out.update(((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)))
    return out


def _trace_outer_loop(cells: set[tuple[int, int]]) -> list[list[int]]:
    """Trace the longest boundary loop of occupied grid cells, as mm vertices."""
    g = _YARD_GRID_MM
    edges: dict[tuple[int, int], tuple[int, int]] = {}
    for (cx, cy) in cells:
        bl = (cx * g, cy * g)
        br = ((cx + 1) * g, cy * g)
        tr = ((cx + 1) * g, (cy + 1) * g)
        tl = (cx * g, (cy + 1) * g)
        if (cx, cy + 1) not in cells:
            edges[tl] = tr
        if (cx + 1, cy) not in cells:
            edges[tr] = br
        if (cx, cy - 1) not in cells:
            edges[br] = bl
        if (cx - 1, cy) not in cells:
            edges[bl] = tl
    best: list[list[int]] = []
    visited: set[tuple[int, int]] = set()
    for start in list(edges):
        if start in visited:
            continue
        loop: list[list[int]] = []
        cur = start
        while cur in edges and cur not in visited:
            visited.add(cur)
            loop.append([cur[0], cur[1]])
            cur = edges[cur]
        if len(loop) > len(best):
            best = loop
    return best


def _grid_boundaries(cells: set[tuple[int, int]]) -> list[list[list[int]]]:
    """Outline polygons enclosing the occupied cells, in millimetres.

    The path is sampled sparsely, so a continuously-mowed region can have small
    gaps; a light morphological closing bridges those. But genuinely separate
    parts of the lawn stay separate: each connected component is traced into its
    OWN polygon, so distinct areas are never joined by a spurious bridge.
    Components smaller than a noise threshold are dropped. Returned largest-first.
    """
    cells = set(cells)
    if len(cells) < _YARD_MIN_COMPONENT_CELLS:
        return []
    work = set(cells)
    for _ in range(_YARD_BOUNDARY_CLOSE_CELLS):
        work = _dilate_cells(work)
    sized: list[tuple[int, list[list[int]]]] = []
    for comp in _connected_components(work):
        original = sum(1 for cell in comp if cell in cells)
        if original < _YARD_MIN_COMPONENT_CELLS:
            continue
        loop = _simplify_collinear(_trace_outer_loop(comp))
        if loop and len(loop) >= 3:
            sized.append((original, loop))
    sized.sort(key=lambda item: item[0], reverse=True)
    return [poly for _, poly in sized]


def _grid_boundary(cells: set[tuple[int, int]]) -> list[list[int]] | None:
    """Largest single boundary polygon (backward-compatible single-polygon view)."""
    polygons = _grid_boundaries(cells)
    return polygons[0] if polygons else None


class AnthbotGenieDataUpdateCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator to fetch and cache Anthbot shadow state."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        account_client: AnthbotCloudApiClient,
        client: AnthbotShadowApiClient,
        device: AnthbotBoundDevice,
        update_interval: timedelta,
    ) -> None:
        super().__init__(
            hass,
            logger=logging.getLogger(__name__),
            name=DOMAIN,
            update_interval=update_interval,
        )
        self.account_client = account_client
        self.client = client
        self.device = device
        # Adaptive polling: the configured interval is the "active" (mowing)
        # cadence; when idle we back off to at least _IDLE_SCAN_INTERVAL so we
        # never poll the shadow endpoint more often than once every 10 min.
        self._active_interval = update_interval
        self._idle_interval = max(update_interval, _IDLE_SCAN_INTERVAL)
        # Monotonic deadline until which polling stays "active" after a command.
        self._force_active_until = 0.0
        # Current 429 back-off interval (None = not rate-limited).
        self._rate_limit_backoff: timedelta | None = None
        self._area_definition: dict[str, Any] = {}
        self._last_area_time: str | None = None
        self._coverage_points: list[list[int]] = []
        self._coverage_mowing = False
        # Persistent yard map (survives restarts, accumulates across sessions).
        self._yard_store: Store[dict[str, Any]] = Store(
            hass, _YARD_STORE_VERSION, f"{DOMAIN}_yard_map_{client.serial_number}"
        )
        self._yard_cells: dict[tuple[int, int], list[int]] = {}
        self._yard_map_id: str | None = None
        self._yard_loaded = False
        self._yard_backup: dict[str, Any] | None = None
        self._yard_pending_map_id: str | None = None
        self._yard_pending_count = 0

    @property
    def reported_state(self) -> dict[str, Any]:
        """Return the latest reported state."""
        return self.data if isinstance(self.data, dict) else {}

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch the latest state from the cloud endpoint."""
        try:
            await self.client.async_ensure_temporary_credentials(self.account_client)
            property_state = await self.client.async_get_shadow_reported_state()
            try:
                service_state = await self.client.async_get_service_reported_state()
            except AnthbotGenieApiError:
                service_state = {}

            # We reached the endpoint successfully -> clear any 429 back-off.
            self._rate_limit_backoff = None

            # Adaptive polling. The mower only streams live telemetry
            # (pose/curpath) to the shadow while a client signals an active app
            # session, and that only matters while it is actually mowing. So:
            #  - mowing (or just after a command from HA, see the grace window)
            #    -> keep the stream alive (app_state) and poll at the configured
            #    fast cadence;
            #  - idle/docked -> send nothing and back off to the slow cadence
            #    (>= 10 min), which keeps us under the shadow endpoint's rate
            #    limit (it returns HTTP 429 TOO_MANY_REQUESTS under frequent polls).
            is_mowing = _raw_robot_status(property_state) in _MOWING_STATES
            forced_active = time.monotonic() < self._force_active_until
            active = is_mowing or forced_active
            if active:
                # Best-effort: never fail the update if the keep-alive command
                # does not go through. Sent during the post-command grace window
                # too, to pre-warm the stream before mowing actually begins.
                try:
                    await self.client.async_publish_service_command(
                        cmd="app_state", data=1
                    )
                except AnthbotGenieApiError as err:
                    self.logger.debug(
                        "Real-time keep-alive (app_state) failed: %s", err
                    )
            desired_interval = (
                self._active_interval if active else self._idle_interval
            )
            if self.update_interval != desired_interval:
                self.update_interval = desired_interval
                self.logger.debug(
                    "Anthbot poll interval -> %s (mowing=%s, forced=%s)",
                    desired_interval,
                    is_mowing,
                    forced_active,
                )

            area_time = property_state.get("area_time")
            if not isinstance(area_time, str):
                area_time = None
            should_refresh_area = not self._area_definition or (
                area_time is not None and area_time != self._last_area_time
            )
            if should_refresh_area:
                try:
                    self._area_definition = (
                        await self.account_client.async_get_device_area_definition(
                            self.client.serial_number
                        )
                    )
                    self._last_area_time = area_time
                except AnthbotGenieApiError:
                    if not self._area_definition:
                        self._area_definition = {}

            self._accumulate_coverage(property_state)
            if not self._yard_loaded:
                await self._async_load_yard_map()
            self._accumulate_yard_map(property_state)

            merged_state = dict(property_state)
            merged_state["_service_reported"] = service_state
            merged_state["_area_definition"] = self._area_definition
            merged_state["_coverage_trail"] = list(self._coverage_points)
            yard_boundaries = _grid_boundaries(set(self._yard_cells))
            merged_state["_yard_map_points"] = list(self._yard_cells.values())
            merged_state["_yard_map_boundaries"] = yard_boundaries
            merged_state["_yard_map_boundary"] = (
                yard_boundaries[0] if yard_boundaries else None
            )
            return merged_state
        except AnthbotGenieApiError as err:
            # If the shadow endpoint rate-limits us (HTTP 429), extend the poll
            # interval exponentially so we stop hammering it; the next successful
            # poll resets this back to the normal active/idle cadence.
            if getattr(err, "status", None) == 429 or "TOO_MANY_REQUESTS" in str(err):
                self._rate_limit_backoff = min(
                    self._rate_limit_backoff * 2
                    if self._rate_limit_backoff
                    else _RATE_LIMIT_BACKOFF_START,
                    _RATE_LIMIT_BACKOFF_MAX,
                )
                self.update_interval = self._rate_limit_backoff
                self.logger.warning(
                    "Anthbot shadow endpoint rate-limited (429); backing off "
                    "to %s",
                    self._rate_limit_backoff,
                )
            raise UpdateFailed(str(err)) from err

    async def async_kick_active_poll(self) -> None:
        """Force fast ("active") polling briefly after a user command.

        A command issued from Home Assistant (e.g. start mowing) should be
        reflected quickly even if the mower was in the slow idle cadence, and we
        must keep polling fast long enough to catch the transition into a mowing
        state (the mower takes a few seconds to start). Opens a short grace
        window during which :meth:`_async_update_data` treats the mower as
        active, switches to the fast interval now, and requests an immediate
        refresh.
        """
        self._force_active_until = time.monotonic() + _COMMAND_ACTIVE_GRACE
        self.update_interval = self._active_interval
        await self.async_request_refresh()

    def _accumulate_coverage(self, property_state: dict[str, Any]) -> None:
        """Accumulate the rolling ``curpath`` window into a growing coverage trail.

        ``curpath`` only carries a sliding ~1 m window of the most recent path,
        so the cumulative trail is built up across polls here: new points are
        appended (consecutive duplicates skipped), the list is reset when a new
        mowing session starts or the mower docks, and it is capped in length.
        """
        status = _raw_robot_status(property_state)
        if status in _MOWING_STATES:
            if not self._coverage_mowing:
                # New mowing session started -> begin a fresh trail.
                self._coverage_points = []
            self._coverage_mowing = True
            for point in _decode_curpath_mm(property_state.get("curpath")):
                if not self._coverage_points or self._coverage_points[-1] != point:
                    self._coverage_points.append(point)
            if len(self._coverage_points) > _COVERAGE_MAX_POINTS:
                self._coverage_points = self._coverage_points[-_COVERAGE_MAX_POINTS:]
        elif status in _DOCK_RESET_STATES:
            self._coverage_points = []
            self._coverage_mowing = False
        # Other states (e.g. paused, unknown): keep the trail unchanged.

    async def _async_load_yard_map(self) -> None:
        """Load the persisted yard map for this mower from disk (once)."""
        self._yard_loaded = True
        try:
            stored = await self._yard_store.async_load()
        except Exception:  # noqa: BLE001
            stored = None
        if isinstance(stored, dict):
            map_id = stored.get("map_id")
            self._yard_map_id = map_id if isinstance(map_id, str) else None
            backup = stored.get("backup")
            self._yard_backup = backup if isinstance(backup, dict) else None
            for point in stored.get("points") or []:
                if (
                    isinstance(point, list)
                    and len(point) == 2
                    and all(isinstance(v, (int, float)) for v in point)
                ):
                    x, y = int(point[0]), int(point[1])
                    self._yard_cells[(x // _YARD_GRID_MM, y // _YARD_GRID_MM)] = [x, y]

    def _yard_map_data(self) -> dict[str, Any]:
        """Serialise the yard map for the Store."""
        data: dict[str, Any] = {
            "map_id": self._yard_map_id,
            "points": list(self._yard_cells.values()),
        }
        if self._yard_backup is not None:
            data["backup"] = self._yard_backup
        return data

    def _accumulate_yard_map(self, property_state: dict[str, Any]) -> None:
        """Accumulate a permanent, cross-session yard map.

        Unlike the session coverage trail this is NOT reset when a mow ends or
        the mower docks. It is auto-reset ONLY when the device reports a real
        ``map_id`` that genuinely differs from the stored one and stays changed
        for several consecutive polls (debounced), to avoid wiping the map on a
        transient reading. When the device reports no ``map_id`` at all, the map
        is never auto-reset (use the manual reset button). Points are deduped
        onto a coarse grid, capped, and persisted so the map fills in 24/7.
        """
        map_id = _device_map_id(property_state)
        changed = False
        if map_id is not None:
            if self._yard_map_id is None:
                # First time we learn a real map id -> adopt it, do NOT reset.
                self._yard_map_id = map_id
                self._yard_pending_map_id = None
                self._yard_pending_count = 0
                changed = True
            elif map_id == self._yard_map_id:
                # Same map -> clear any pending re-map candidate.
                self._yard_pending_map_id = None
                self._yard_pending_count = 0
            else:
                # A different real map id -> candidate re-map; require it to be
                # stable for several polls before wiping (debounce).
                if map_id == self._yard_pending_map_id:
                    self._yard_pending_count += 1
                else:
                    self._yard_pending_map_id = map_id
                    self._yard_pending_count = 1
                if self._yard_pending_count >= _YARD_REMAP_CONFIRM_POLLS:
                    # Confirmed re-map: back up the old map, then start fresh.
                    if self._yard_cells:
                        self._yard_backup = {
                            "map_id": self._yard_map_id,
                            "points": list(self._yard_cells.values()),
                        }
                    self._yard_cells = {}
                    self._yard_map_id = map_id
                    self._yard_pending_map_id = None
                    self._yard_pending_count = 0
                    changed = True
        # When map_id is None this poll, the identity is left untouched and the
        # map is never reset.

        if _raw_robot_status(property_state) in _MOWING_STATES:
            points = _decode_curpath_mm(property_state.get("curpath"))
            pose = property_state.get("pose")
            if isinstance(pose, dict):
                px, py = pose.get("x"), pose.get("y")
                if isinstance(px, (int, float)) and isinstance(py, (int, float)):
                    points.append([int(px), int(py)])
            for x, y in points:
                cell = (x // _YARD_GRID_MM, y // _YARD_GRID_MM)
                if cell not in self._yard_cells:
                    if len(self._yard_cells) >= _YARD_MAX_CELLS:
                        continue
                    self._yard_cells[cell] = [x, y]
                    changed = True

        if changed:
            self._yard_store.async_delay_save(self._yard_map_data, _YARD_SAVE_DELAY)

    async def async_reset_yard_map(self) -> None:
        """Manually clear the persistent yard map, keeping a backup for recovery."""
        if not self._yard_loaded:
            await self._async_load_yard_map()
        if self._yard_cells:
            self._yard_backup = {
                "map_id": self._yard_map_id,
                "points": list(self._yard_cells.values()),
            }
        self._yard_cells = {}
        self._yard_pending_map_id = None
        self._yard_pending_count = 0
        await self._yard_store.async_save(self._yard_map_data())
        await self.async_request_refresh()
