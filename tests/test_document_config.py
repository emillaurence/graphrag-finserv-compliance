"""
Unit tests for src/document/config.py
"""

import pytest
from unittest.mock import mock_open, patch
from pathlib import Path

from src.document.config import load_document_config


class TestLoadDocumentConfig:
    def test_loads_yaml_file(self):
        yaml_content = "documents:\n  - name: APS-112\n    path: aps_112.pdf\n"
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            result = load_document_config(Path("config.yaml"))

        assert result == {"documents": [{"name": "APS-112", "path": "aps_112.pdf"}]}

    def test_returns_dict(self):
        yaml_content = "key: value\n"
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            result = load_document_config(Path("config.yaml"))

        assert isinstance(result, dict)
        assert result["key"] == "value"

    def test_handles_empty_yaml(self):
        with patch("builtins.open", mock_open(read_data="")):
            result = load_document_config(Path("config.yaml"))

        assert result is None
