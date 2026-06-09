"""Data coordinator for Anthbot Genie."""

from __future__ import annotations

import base64
from datetime import timedelta
import logging
import struct
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
_YARD_GRID_MM = 200  # dedup grid cell size in millimetres
_YARD_MAX_CELLS = 5000
_YARD_SAVE_DELAY = 30  # seconds to batch Store writes
_YARD_STORE_VERSION = 1


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


def _map_identity(data: dict[str, Any]) -> str | None:
    """Return a stable identifier for the device's current map.

    Prefers ``multi_maps.map_list[0].map_id``; falls back to the map version
    timestamps when that is absent (e.g. some Genie firmwares report an empty
    ``multi_maps``). Used only to detect a re-map (which resets the yard map).
    """
    multi = data.get("multi_maps")
    if isinstance(multi, dict):
        map_list = multi.get("map_list")
        if isinstance(map_list, list) and map_list and isinstance(map_list[0], dict):
            map_id = map_list[0].get("map_id")
            if isinstance(map_id, str) and map_id:
                return map_id
    for key in ("map_time", "map_tar_time", "area_time"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
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


_YARD_BOUNDARY_MAX_DILATE = 8  # bridge gaps up to ~1.6 m to merge sparse fragments


def _convex_hull(points: list[list[int]]) -> list[list[int]] | None:
    """Convex hull (monotone chain) of points in mm; always encloses every point."""
    pts = sorted({(int(p[0]), int(p[1])) for p in points})
    if len(pts) < 3:
        return None

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[int, int]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[int, int]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return [[x, y] for x, y in hull] if len(hull) >= 3 else None


def _cells_single_component(cells: set[tuple[int, int]]) -> bool:
    """Whether the occupied cells form a single 4-connected component."""
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        cx, cy = stack.pop()
        for nb in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


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


def _grid_boundary(cells: set[tuple[int, int]]) -> list[list[int]] | None:
    """Outline polygon enclosing ALL occupied cells, in millimetres.

    The path is sampled sparsely (a short window per poll), so the occupied
    cells usually form several disconnected fragments. Tracing the grid directly
    would return only the largest fragment — correct shape but several times too
    small. Instead, morphologically dilate the cells until they form a single
    connected region, then trace that region's outer loop: this keeps the
    concave grid shape while enclosing everything. If the fragments are too far
    apart to bridge within the cap, fall back to a convex hull of all cells
    (which always encloses every point).
    """
    if len(cells) < 8:
        return None
    g = _YARD_GRID_MM
    work = set(cells)
    dilations = 0
    while not _cells_single_component(work) and dilations < _YARD_BOUNDARY_MAX_DILATE:
        work = _dilate_cells(work)
        dilations += 1
    if not _cells_single_component(work):
        return _convex_hull([[cx * g + g // 2, cy * g + g // 2] for cx, cy in cells])
    simplified = _simplify_collinear(_trace_outer_loop(work))
    if simplified:
        return simplified
    return _convex_hull([[cx * g + g // 2, cy * g + g // 2] for cx, cy in cells])


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

    @property
    def reported_state(self) -> dict[str, Any]:
        """Return the latest reported state."""
        return self.data if isinstance(self.data, dict) else {}

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch the latest state from the cloud endpoint."""
        try:
            await self.client.async_ensure_temporary_credentials(self.account_client)
            # Keep the device's real-time stream alive. The mower only streams
            # live telemetry (pose/curpath) to the cloud shadow while a client
            # signals an active app session; it stops roughly 60 s after the
            # last signal. Re-sending it every poll keeps pose/curpath fresh in
            # HA without the phone app open. Best-effort: never fail the update
            # if the keep-alive command does not go through.
            try:
                await self.client.async_publish_service_command(
                    cmd="app_state", data=1
                )
            except AnthbotGenieApiError as err:
                self.logger.debug("Real-time keep-alive (app_state) failed: %s", err)
            property_state = await self.client.async_get_shadow_reported_state()
            try:
                service_state = await self.client.async_get_service_reported_state()
            except AnthbotGenieApiError:
                service_state = {}

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
            merged_state["_yard_map_points"] = list(self._yard_cells.values())
            merged_state["_yard_map_boundary"] = _grid_boundary(set(self._yard_cells))
            return merged_state
        except AnthbotGenieApiError as err:
            raise UpdateFailed(str(err)) from err

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
        return {
            "map_id": self._yard_map_id,
            "points": list(self._yard_cells.values()),
        }

    def _accumulate_yard_map(self, property_state: dict[str, Any]) -> None:
        """Accumulate a permanent, cross-session yard map.

        Unlike the session coverage trail this is NOT reset when a mow ends or
        the mower docks; it only resets when the device's map identity changes
        (a re-map). Points are deduped onto a coarse grid, capped, and persisted
        to disk so the map keeps filling in 24/7 and survives restarts.
        """
        map_id = _map_identity(property_state)
        changed = False
        if (
            map_id is not None
            and self._yard_map_id is not None
            and map_id != self._yard_map_id
        ):
            # The device map was replaced -> start a fresh yard map.
            self._yard_cells = {}
            changed = True
        if map_id is not None and map_id != self._yard_map_id:
            self._yard_map_id = map_id
            changed = True

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
