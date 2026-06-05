"""
Unit tests for src/document/pdf_utils.py

pdfplumber is mocked — no PDF files required.
"""

import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from src.document.pdf_utils import extract_pdf_pages, batch_to_text, extract_full_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_pdf(page_texts: list[str]):
    """Return a mock pdfplumber.open() context that yields pages with given texts."""
    pages = []
    for text in page_texts:
        page = MagicMock()
        page.extract_text.return_value = text
        pages.append(page)

    mock_pdf = MagicMock()
    mock_pdf.pages = pages
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    return mock_pdf


# ---------------------------------------------------------------------------
# extract_pdf_pages
# ---------------------------------------------------------------------------


class TestExtractPdfPages:
    @patch("src.document.pdf_utils.pdfplumber.open")
    def test_returns_page_tuples(self, mock_open_pdf):
        mock_open_pdf.return_value = _make_mock_pdf(["Page 1 text", "Page 2 text"])

        pages = extract_pdf_pages(Path("dummy.pdf"))

        assert len(pages) == 2
        assert pages[0] == (1, "Page 1 text")
        assert pages[1] == (2, "Page 2 text")

    @patch("src.document.pdf_utils.pdfplumber.open")
    def test_handles_empty_page(self, mock_open_pdf):
        mock_open_pdf.return_value = _make_mock_pdf([None])

        pages = extract_pdf_pages(Path("dummy.pdf"))

        assert pages == [(1, "")]

    @patch("src.document.pdf_utils.pdfplumber.open")
    def test_handles_empty_pdf(self, mock_open_pdf):
        mock_open_pdf.return_value = _make_mock_pdf([])

        pages = extract_pdf_pages(Path("dummy.pdf"))

        assert pages == []


# ---------------------------------------------------------------------------
# batch_to_text
# ---------------------------------------------------------------------------


class TestBatchToText:
    def test_joins_with_page_markers(self):
        batch = [(1, "Hello"), (2, "World")]
        result = batch_to_text(batch)

        assert "--- PAGE 1 ---" in result
        assert "Hello" in result
        assert "--- PAGE 2 ---" in result
        assert "World" in result

    def test_empty_batch(self):
        assert batch_to_text([]) == ""

    def test_single_page(self):
        result = batch_to_text([(5, "content")])
        assert "--- PAGE 5 ---" in result
        assert "content" in result


# ---------------------------------------------------------------------------
# extract_full_text
# ---------------------------------------------------------------------------


class TestExtractFullText:
    @patch("src.document.pdf_utils.pdfplumber.open")
    def test_concatenates_all_pages(self, mock_open_pdf):
        mock_open_pdf.return_value = _make_mock_pdf(["First", "Second", "Third"])

        result = extract_full_text(Path("dummy.pdf"))

        assert "First" in result
        assert "Second" in result
        assert "Third" in result
        assert result == "First\n\nSecond\n\nThird"

    @patch("src.document.pdf_utils.pdfplumber.open")
    def test_handles_none_pages(self, mock_open_pdf):
        mock_open_pdf.return_value = _make_mock_pdf([None, "real text", None])

        result = extract_full_text(Path("dummy.pdf"))

        assert "real text" in result
