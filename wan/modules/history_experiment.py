"""Bind JSON experiment configuration to the parameter-free KV interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .history_layout import build_history_layout_planner
from .kv_memory import (
    ChunkLayoutHistoryPolicy,
    NoMemoryManagementPolicy,
    SinkRecentHistoryPolicy,
)


def load_history_policy_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("history policy config must be a JSON object")
    return config


def configure_history_policy(model, config: Mapping[str, Any]) -> None:
    """Configure every attention block without changing model parameters."""

    policy_type = config.get("policy")
    for block in model.blocks:
        attention = block.self_attn
        if policy_type == "official_sink_recent":
            attention.history_policy = SinkRecentHistoryPolicy(
                attention.local_attn_size,
                attention.max_attention_size,
                attention.sink_size,
            )
        elif policy_type == "no_memory_management":
            attention.history_policy = NoMemoryManagementPolicy()
        elif policy_type == "chunk_layout":
            # Each layer receives its own stateless planner instance so a future
            # planner may safely gain layer-local instrumentation.
            planner = build_history_layout_planner(config["planner"])
            attention.history_policy = ChunkLayoutHistoryPolicy(planner)
        else:
            raise ValueError(f"Unknown history policy: {policy_type!r}")


def required_cache_frames(config: Mapping[str, Any], output_frames: int) -> int | None:
    """Return a physical cache override for append-only experiment policies."""

    if config.get("policy") in {"chunk_layout", "no_memory_management"}:
        configured = config.get("cache_frames", output_frames)
        if int(configured) < output_frames:
            raise ValueError(
                f"cache_frames={configured} is smaller than output_frames={output_frames}"
            )
        return int(configured)
    return None
