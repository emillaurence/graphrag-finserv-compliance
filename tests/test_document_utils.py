"""
Unit tests for src/document/utils.py

Covers:
  - strip_fences: removes markdown code fences
  - serialise_row: converts nested dicts/lists to JSON strings for CSV
  - call_claude_stream: streaming Claude call (mocked)
  - call_claude_stream_json: streaming + JSON parse + retry (mocked)
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.document.utils import strip_fences, serialise_row, call_claude_stream, call_claude_stream_json


# ---------------------------------------------------------------------------
# strip_fences
# ---------------------------------------------------------------------------


class TestStripFences:
    def test_removes_json_fences(self):
        text = '```json\n{"key": "value"}\n```'
        assert strip_fences(text) == '{"key": "value"}'

    def test_removes_plain_fences(self):
        text = '```\nhello world\n```'
        assert strip_fences(text) == "hello world"

    def test_preserves_unfenced_text(self):
        text = '{"key": "value"}'
        assert strip_fences(text) == '{"key": "value"}'

    def test_strips_surrounding_whitespace(self):
        text = '  \n```\nfoo\n```\n  '
        assert strip_fences(text) == "foo"

    def test_preserves_inner_backticks(self):
        text = '```\nsome `code` here\n```'
        assert strip_fences(text) == "some `code` here"

    def test_handles_empty_string(self):
        assert strip_fences("") == ""

    def test_handles_only_fences(self):
        text = "```\n```"
        assert strip_fences(text) == ""

    def test_multiline_content(self):
        text = '```python\nline1\nline2\nline3\n```'
        assert strip_fences(text) == "line1\nline2\nline3"


# ---------------------------------------------------------------------------
# serialise_row
# ---------------------------------------------------------------------------


class TestSerialiseRow:
    def test_nested_dict_becomes_json_string(self):
        row = {"name": "Acme", "meta": {"country": "AU"}}
        out = serialise_row(row)
        assert out["name"] == "Acme"
        assert out["meta"] == json.dumps({"country": "AU"})

    def test_nested_list_becomes_json_string(self):
        row = {"ids": [1, 2, 3]}
        out = serialise_row(row)
        assert out["ids"] == "[1, 2, 3]"

    def test_scalars_unchanged(self):
        row = {"a": 1, "b": "hello", "c": 3.14, "d": True, "e": None}
        out = serialise_row(row)
        assert out == row

    def test_empty_row(self):
        assert serialise_row({}) == {}

    def test_mixed_types(self):
        row = {"x": "plain", "y": {"nested": True}, "z": [1]}
        out = serialise_row(row)
        assert out["x"] == "plain"
        assert isinstance(out["y"], str)
        assert isinstance(out["z"], str)


# ---------------------------------------------------------------------------
# call_claude_stream
# ---------------------------------------------------------------------------


class TestCallClaudeStream:
    def test_returns_response_text(self):
        mock_block = MagicMock()
        mock_block.text = "Generated text"

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [mock_block]

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.get_final_message.return_value = mock_response

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream

        result = call_claude_stream(
            mock_client, "claude-sonnet-4-6", 4096, "system prompt",
            [{"role": "user", "content": "hello"}],
        )
        assert result == "Generated text"

    def test_raises_on_max_tokens_truncation(self):
        mock_block = MagicMock()
        mock_block.text = "Truncated..."

        mock_response = MagicMock()
        mock_response.stop_reason = "max_tokens"
        mock_response.content = [mock_block]

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.get_final_message.return_value = mock_response

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream

        with pytest.raises(RuntimeError, match="truncated"):
            call_claude_stream(
                mock_client, "claude-sonnet-4-6", 1000, "system",
                [{"role": "user", "content": "x"}],
            )


# ---------------------------------------------------------------------------
# call_claude_stream_json
# ---------------------------------------------------------------------------


class TestCallClaudeStreamJson:
    def _make_stream_mock(self, client, text, stop_reason="end_turn"):
        mock_block = MagicMock()
        mock_block.text = text
        mock_response = MagicMock()
        mock_response.stop_reason = stop_reason
        mock_response.content = [mock_block]
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.get_final_message.return_value = mock_response
        client.messages.stream.return_value = mock_stream

    def test_parses_valid_json(self):
        client = MagicMock()
        self._make_stream_mock(client, '{"key": "value"}')

        result = call_claude_stream_json(
            client, "model", 4096, "system",
            [{"role": "user", "content": "x"}],
        )
        assert result == {"key": "value"}

    def test_strips_fences_before_parsing(self):
        client = MagicMock()
        self._make_stream_mock(client, '```json\n{"a": 1}\n```')

        result = call_claude_stream_json(
            client, "model", 4096, "system",
            [{"role": "user", "content": "x"}],
        )
        assert result == {"a": 1}

    @patch("src.document.utils.time.sleep")
    def test_retries_on_json_error(self, mock_sleep):
        client = MagicMock()

        # First call returns invalid JSON, second returns valid
        bad_block = MagicMock()
        bad_block.text = "not json {"
        bad_response = MagicMock()
        bad_response.stop_reason = "end_turn"
        bad_response.content = [bad_block]

        good_block = MagicMock()
        good_block.text = '{"fixed": true}'
        good_response = MagicMock()
        good_response.stop_reason = "end_turn"
        good_response.content = [good_block]

        mock_stream_bad = MagicMock()
        mock_stream_bad.__enter__ = MagicMock(return_value=mock_stream_bad)
        mock_stream_bad.__exit__ = MagicMock(return_value=False)
        mock_stream_bad.get_final_message.return_value = bad_response

        mock_stream_good = MagicMock()
        mock_stream_good.__enter__ = MagicMock(return_value=mock_stream_good)
        mock_stream_good.__exit__ = MagicMock(return_value=False)
        mock_stream_good.get_final_message.return_value = good_response

        client.messages.stream.side_effect = [mock_stream_bad, mock_stream_good]

        result = call_claude_stream_json(
            client, "model", 4096, "system",
            [{"role": "user", "content": "x"}],
        )
        assert result == {"fixed": True}
        mock_sleep.assert_called_once_with(2)
