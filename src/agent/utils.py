"""Shared utilities for all agent classes."""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import anthropic

from src.agent._security import guard_tool_result
from src.agent.config import MAX_RETRY_SECONDS, TOOL_RESULT_CHAR_LIMIT

logger = logging.getLogger(__name__)

ENTITY_ID_RE = re.compile(r"(BRW|LOAN|ACC|TXN)-\d+", re.IGNORECASE)

ENTITY_PREFIX_TO_TYPE: dict[str, str] = {
    "LOAN": "LoanApplication",
    "BRW": "Borrower",
    "ACC": "BankAccount",
    "TXN": "Transaction",
}


def clean_markdown(s: str) -> str:
    """Strip markdown bold/italic markers and surrounding whitespace."""
    return s.strip().strip("*").strip()


def call_claude_with_retry(
    client: anthropic.Anthropic, *, label: str = "", **kwargs: Any
) -> anthropic.types.Message:
    """Call client.messages.create with up to 3 attempts on RateLimitError.

    Reads the retry-after response header when available; falls back to
    capped exponential backoff (30s, 60s).

    Logs one INFO line per successful call:
        <model> <label|stop_reason> <elapsed>s | in=N out=N cached=N
    """
    for attempt in range(3):
        try:
            t0 = time.perf_counter()
            response = client.messages.create(**kwargs)
            elapsed = time.perf_counter() - t0
            usage = response.usage
            cached  = getattr(usage, "cache_read_input_tokens",    0) or 0
            created = getattr(usage, "cache_creation_input_tokens", 0) or 0
            tag = label or response.stop_reason or ""
            logger.info(
                "%s %s %.2fs | in=%d out=%d cached=%d created=%d",
                response.model, tag, elapsed,
                usage.input_tokens, usage.output_tokens, cached, created,
            )
            return response
        except anthropic.RateLimitError as e:
            if attempt < 2:
                retry_after = None
                try:
                    h = getattr(e, "response", None) and getattr(e.response, "headers", None)
                    if h:
                        retry_after = h.get("retry-after")
                        if retry_after is not None:
                            retry_after = min(int(float(retry_after)), MAX_RETRY_SECONDS)
                except (TypeError, ValueError):
                    pass
                wait = retry_after if retry_after is not None else min(30 * (2 ** attempt), MAX_RETRY_SECONDS)
                logger.warning("Rate limited — waiting %ds (attempt %d/3)", wait, attempt + 1)
                time.sleep(wait)
            else:
                raise


def truncate_tool_result(content: str, limit: int = TOOL_RESULT_CHAR_LIMIT) -> str:
    """Truncate a tool result string to `limit` characters (default TOOL_RESULT_CHAR_LIMIT)."""
    if len(content) > limit:
        return content[:limit] + "… [truncated]"
    return content


def extract_field(text: str, pattern: str, default: str = "") -> str:
    """Return the first capture group of pattern matched against text, or default."""
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else default


def extract_text(response: anthropic.types.Message) -> str:
    """Return the first text block from a Claude response, or empty string."""
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return ""


def trim_message_history(
    messages: list[dict], max_pairs: int, anchor_count: int = 1
) -> list[dict]:
    """Trim message history to at most max_pairs tool-use/tool-result round-trips.

    Always preserves messages[:anchor_count] (default: the initial user question).
    When anchor_count > 1 (e.g. pre-run traverse results injected before the loop),
    those messages are anchored and the sliding window applies only to the tail.
    Drops orphaned tool_result blocks that lost their assistant/tool_use pair.
    """
    anchor = messages[:anchor_count]
    tail = messages[anchor_count:]
    max_tail_msgs = max_pairs * 2
    if len(tail) <= max_tail_msgs:
        return anchor + tail
    trimmed = tail[-(max_tail_msgs):]
    if trimmed[0].get("role") == "user":
        trimmed = trimmed[1:]
    return anchor + trimmed


# ---------------------------------------------------------------------------
# Entity parsing
# ---------------------------------------------------------------------------


def parse_entity_from_question(question: str) -> tuple[str, str]:
    """Extract the first entity ID and its type from a natural-language question.

    Returns (entity_id, entity_type) — both empty strings when no ID is found.
    """
    match = ENTITY_ID_RE.search(question)
    if not match:
        return "", ""
    entity_id = match.group(0).upper()
    prefix = entity_id.split("-")[0]
    entity_type = ENTITY_PREFIX_TO_TYPE.get(prefix, "")
    return entity_id, entity_type


# ---------------------------------------------------------------------------
# Parallel tool execution
# ---------------------------------------------------------------------------


def execute_tools_parallel(
    execute_tool_fn: Any,
    tool_blocks: list,
) -> dict[str, dict]:
    """Execute tool-use blocks in parallel and return {block.id: result}.

    Each entry in *tool_blocks* must expose `.id`, `.name`, and `.input`
    (i.e. Anthropic SDK ``ToolUseBlock`` objects or compatible dicts).
    """
    results_map: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(tool_blocks))) as ex:
        future_to_id = {
            ex.submit(execute_tool_fn, blk.name, blk.input): blk.id
            for blk in tool_blocks
        }
        for future in as_completed(future_to_id):
            results_map[future_to_id[future]] = future.result()
    return results_map


# ---------------------------------------------------------------------------
# Tool result processing
# ---------------------------------------------------------------------------


def process_tool_result(
    result: dict,
    tool_name: str,
    limit: int = TOOL_RESULT_CHAR_LIMIT,
) -> str:
    """Serialise, truncate, and apply security framing to a tool result."""
    content = truncate_tool_result(json.dumps(result, default=str), limit)
    return guard_tool_result(content, tool_name)


def build_tool_result_message(tool_use_id: str, content: str) -> dict:
    """Build a single ``tool_result`` content block for the messages list."""
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }


# ---------------------------------------------------------------------------
# Anthropic content-block helpers
# ---------------------------------------------------------------------------


def blocks_to_dicts(content: Any) -> list[dict]:
    """Convert Anthropic SDK content blocks to plain dicts for stable serialisation."""
    result = []
    for block in content:
        if isinstance(block, dict):
            result.append(block)
        elif block.type == "text":
            result.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            result.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
    return result
