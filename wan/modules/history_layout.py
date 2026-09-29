"""Pure-Python planners for experimental historical-chunk layouts.

The planners deliberately know nothing about PyTorch or KV tensors.  They
receive the chunks that have been retained by a store and return the ordered
chunk ids that attention should see.  Keeping this layer pure makes manifest
validation and layout tests runnable on a development laptop without the
LiveEdit CUDA environment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ChunkSpan:
    """One append unit in the causal KV stream."""

    chunk_id: int
    frame_start: int
    frame_count: int
    token_start: int
    token_count: int


class HistoryLayoutPlanner(ABC):
    """Choose an ordered subset of retained chunks, including current."""

    @abstractmethod
    def select(self, chunks: Sequence[ChunkSpan]) -> list[int]:
        """Return chunk ids in the exact order used by attention."""


def _dedupe(ids: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for chunk_id in ids:
        if chunk_id not in seen:
            result.append(chunk_id)
            seen.add(chunk_id)
    return result


def _finalize(
    selected: Iterable[int],
    chunks: Sequence[ChunkSpan],
    order: str,
) -> list[int]:
    if not chunks:
        return []
    available = {chunk.chunk_id for chunk in chunks}
    current = chunks[-1].chunk_id
    result = [chunk_id for chunk_id in _dedupe(selected) if chunk_id in available]
    if current not in result:
        result.append(current)
    if order == "chronological":
        result.sort()
    elif order == "reverse_chronological":
        result.sort(reverse=True)
    elif order != "selection":
        raise ValueError(
            "order must be chronological, reverse_chronological, or selection, "
            f"got {order!r}"
        )
    # Only historical chunks are reorderable. Q belongs to the current chunk,
    # so current KV must remain at the final logical position for window RoPE.
    result.remove(current)
    result.append(current)
    return result


class AllHistoryPlanner(HistoryLayoutPlanner):
    """Policy-free baseline: expose every retained chunk."""

    def select(self, chunks: Sequence[ChunkSpan]) -> list[int]:
        return [chunk.chunk_id for chunk in chunks]


class SinkRecentPlanner(HistoryLayoutPlanner):
    """Chunk-level first-N sink plus newest-M layout."""

    def __init__(self, sink_count: int, recent_count: int):
        if sink_count < 0 or recent_count < 0:
            raise ValueError("sink_count and recent_count must be non-negative")
        self.sink_count = sink_count
        self.recent_count = recent_count

    def select(self, chunks: Sequence[ChunkSpan]) -> list[int]:
        if not chunks:
            return []
        sink = chunks[: self.sink_count]
        recent = chunks[-self.recent_count :] if self.recent_count else []
        return _finalize(
            [chunk.chunk_id for chunk in (*sink, *recent)],
            chunks,
            "chronological",
        )


class ExplicitLayoutPlanner(HistoryLayoutPlanner):
    """Use a per-current-chunk schedule, useful for controlled ablations."""

    def __init__(self, schedule: Mapping[str, Sequence[int]], order: str = "selection"):
        self.schedule = {str(key): list(value) for key, value in schedule.items()}
        self.order = order

    def select(self, chunks: Sequence[ChunkSpan]) -> list[int]:
        if not chunks:
            return []
        current = chunks[-1].chunk_id
        selected = self.schedule.get(str(current), self.schedule.get("default", []))
        return _finalize(selected, chunks, self.order)


class AnchorRecentPlanner(HistoryLayoutPlanner):
    """Use fixed or scheduled anchors instead of always anchoring at the start."""

    def __init__(
        self,
        anchors: Sequence[int],
        recent_count: int,
        schedule: Mapping[str, Sequence[int]] | None = None,
        order: str = "chronological",
    ):
        if recent_count < 0:
            raise ValueError("recent_count must be non-negative")
        self.anchors = list(anchors)
        self.recent_count = recent_count
        self.schedule = {
            str(key): list(value) for key, value in (schedule or {}).items()
        }
        self.order = order

    def select(self, chunks: Sequence[ChunkSpan]) -> list[int]:
        if not chunks:
            return []
        current = chunks[-1].chunk_id
        anchors = self.schedule.get(str(current), self.anchors)
        recent = chunks[-self.recent_count :] if self.recent_count else []
        return _finalize(
            [*anchors, *(chunk.chunk_id for chunk in recent)],
            chunks,
            self.order,
        )


class StridedHistoryPlanner(HistoryLayoutPlanner):
    """Sample non-contiguous historical chunks at a fixed stride."""

    def __init__(
        self,
        stride: int,
        recent_count: int = 1,
        offset: int = 0,
        max_history: int | None = None,
        order: str = "chronological",
    ):
        if stride <= 0:
            raise ValueError("stride must be positive")
        if recent_count < 0:
            raise ValueError("recent_count must be non-negative")
        self.stride = stride
        self.recent_count = recent_count
        self.offset = offset
        self.max_history = max_history
        self.order = order

    def select(self, chunks: Sequence[ChunkSpan]) -> list[int]:
        if not chunks:
            return []
        past = [
            chunk.chunk_id
            for chunk in chunks[:-1]
            if (chunk.chunk_id - self.offset) % self.stride == 0
        ]
        if self.max_history is not None:
            past = past[-self.max_history :]
        recent = chunks[-self.recent_count :] if self.recent_count else []
        return _finalize(
            [*past, *(chunk.chunk_id for chunk in recent)],
            chunks,
            self.order,
        )


class SceneContrastPlanner(HistoryLayoutPlanner):
    """Combine recent context with chunks carrying a different scene label."""

    def __init__(
        self,
        scene_by_chunk: Mapping[str, str],
        contrast_count: int,
        recent_count: int = 1,
        scene_distance: Mapping[str, float] | None = None,
        order: str = "chronological",
    ):
        if contrast_count < 0 or recent_count < 0:
            raise ValueError("contrast_count and recent_count must be non-negative")
        self.scene_by_chunk = {int(key): value for key, value in scene_by_chunk.items()}
        self.contrast_count = contrast_count
        self.recent_count = recent_count
        self.scene_distance = dict(scene_distance or {})
        self.order = order

    def _distance(self, left: str | None, right: str | None) -> float:
        if left is None or right is None or left == right:
            return 0.0
        return float(
            self.scene_distance.get(
                f"{left}|{right}", self.scene_distance.get(f"{right}|{left}", 1.0)
            )
        )

    def select(self, chunks: Sequence[ChunkSpan]) -> list[int]:
        if not chunks:
            return []
        current = chunks[-1].chunk_id
        current_scene = self.scene_by_chunk.get(current)
        ranked = sorted(
            (
                (
                    self._distance(
                        self.scene_by_chunk.get(chunk.chunk_id), current_scene
                    ),
                    chunk.chunk_id,
                )
                for chunk in chunks[:-1]
            ),
            reverse=True,
        )
        contrast = [
            chunk_id for distance, chunk_id in ranked[: self.contrast_count]
            if distance > 0
        ]
        recent = chunks[-self.recent_count :] if self.recent_count else []
        return _finalize(
            [*contrast, *(chunk.chunk_id for chunk in recent)],
            chunks,
            self.order,
        )


def build_history_layout_planner(config: Mapping[str, Any]) -> HistoryLayoutPlanner:
    """Construct a planner from a JSON-compatible configuration."""

    planner_type = config.get("type")
    if planner_type == "all":
        return AllHistoryPlanner()
    if planner_type == "sink_recent":
        return SinkRecentPlanner(
            sink_count=int(config.get("sink_count", 0)),
            recent_count=int(config.get("recent_count", 0)),
        )
    if planner_type == "explicit":
        return ExplicitLayoutPlanner(
            schedule=config.get("schedule", {}),
            order=config.get("order", "selection"),
        )
    if planner_type == "anchor_recent":
        return AnchorRecentPlanner(
            anchors=config.get("anchors", []),
            recent_count=int(config.get("recent_count", 1)),
            schedule=config.get("schedule"),
            order=config.get("order", "chronological"),
        )
    if planner_type == "stride":
        max_history = config.get("max_history")
        return StridedHistoryPlanner(
            stride=int(config["stride"]),
            recent_count=int(config.get("recent_count", 1)),
            offset=int(config.get("offset", 0)),
            max_history=int(max_history) if max_history is not None else None,
            order=config.get("order", "chronological"),
        )
    if planner_type == "scene_contrast":
        return SceneContrastPlanner(
            scene_by_chunk=config.get("scene_by_chunk", {}),
            contrast_count=int(config.get("contrast_count", 1)),
            recent_count=int(config.get("recent_count", 1)),
            scene_distance=config.get("scene_distance"),
            order=config.get("order", "chronological"),
        )
    raise ValueError(f"Unknown history layout planner type: {planner_type!r}")
