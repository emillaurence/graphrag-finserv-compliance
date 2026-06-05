"""
Unit tests for src/agent/dispatcher.py

Neo4j connection and tool_impl functions are fully mocked.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.agent.dispatcher import make_execute_tool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_conn():
    conn = MagicMock()
    conn.run_query.return_value = [{"n": 1}]
    return conn


# ---------------------------------------------------------------------------
# read-neo4j-cypher routing
# ---------------------------------------------------------------------------


class TestReadNeo4jCypher:
    def test_executes_read_query(self, mock_conn):
        execute_tool = make_execute_tool(mock_conn)
        result = execute_tool("read-neo4j-cypher", {"query": "MATCH (n) RETURN n LIMIT 1"})

        assert result == {"rows": [{"n": 1}]}
        mock_conn.run_query.assert_called_once_with(
            "MATCH (n) RETURN n LIMIT 1", {}
        )

    def test_passes_params(self, mock_conn):
        execute_tool = make_execute_tool(mock_conn)
        execute_tool(
            "read-neo4j-cypher",
            {"query": "MATCH (n {id: $id}) RETURN n", "params": {"id": "X"}},
        )
        mock_conn.run_query.assert_called_once_with(
            "MATCH (n {id: $id}) RETURN n", {"id": "X"}
        )

    def test_blocks_write_keywords(self, mock_conn):
        execute_tool = make_execute_tool(mock_conn)

        for keyword in ["MERGE", "CREATE", "DELETE", "SET", "DETACH", "REMOVE", "DROP"]:
            result = execute_tool(
                "read-neo4j-cypher",
                {"query": f"{keyword} (n:Node) RETURN n"},
            )
            assert "error" in result, f"Expected error for write keyword '{keyword}'"
            assert "write" in result["error"].lower()

    def test_allows_set_inside_word(self, mock_conn):
        """ASSESSMENT contains SET but should not be blocked (whole-word match)."""
        execute_tool = make_execute_tool(mock_conn)
        result = execute_tool(
            "read-neo4j-cypher",
            {"query": "MATCH (a:Assessment) RETURN a LIMIT 10"},
        )
        assert "rows" in result


# ---------------------------------------------------------------------------
# write-neo4j-cypher routing
# ---------------------------------------------------------------------------


class TestWriteNeo4jCypher:
    def test_executes_write_query(self, mock_conn):
        execute_tool = make_execute_tool(mock_conn)
        result = execute_tool(
            "write-neo4j-cypher",
            {"query": "MERGE (a:Assessment {id: 'X'}) SET a.verdict = 'COMPLIANT'"},
        )
        assert result == {"rows": [{"n": 1}]}


# ---------------------------------------------------------------------------
# FastMCP tool routing
# ---------------------------------------------------------------------------


class TestFastMCPRouting:
    @patch("src.agent.dispatcher.traverse_compliance_path")
    def test_routes_traverse_compliance_path(self, mock_fn, mock_conn):
        mock_fn.return_value = {"entity": {}, "regulations": {}}
        execute_tool = make_execute_tool(mock_conn)

        result = execute_tool(
            "traverse_compliance_path",
            {"entity_id": "LOAN-0002", "entity_type": "LoanApplication"},
        )

        mock_fn.assert_called_once_with(
            entity_id="LOAN-0002", entity_type="LoanApplication", conn=mock_conn
        )
        assert result == {"entity": {}, "regulations": {}}

    @patch("src.agent.dispatcher.detect_graph_anomalies")
    def test_routes_detect_graph_anomalies(self, mock_fn, mock_conn):
        mock_fn.return_value = {"patterns_run": 0, "total_findings": 0, "results": []}
        execute_tool = make_execute_tool(mock_conn)

        result = execute_tool(
            "detect_graph_anomalies",
            {"pattern_names": ["high_lvr_loans"]},
        )

        mock_fn.assert_called_once_with(
            pattern_names=["high_lvr_loans"], conn=mock_conn
        )

    @patch("src.agent.dispatcher.retrieve_regulatory_chunks")
    def test_routes_retrieve_regulatory_chunks(self, mock_fn, mock_conn):
        mock_fn.return_value = {"query": "test", "chunks": []}
        execute_tool = make_execute_tool(mock_conn)

        execute_tool(
            "retrieve_regulatory_chunks",
            {"query_text": "LVR limits"},
        )

        mock_fn.assert_called_once_with(query_text="LVR limits", conn=mock_conn)

    @patch("src.agent.dispatcher.persist_assessment")
    def test_routes_persist_assessment(self, mock_fn, mock_conn):
        mock_fn.return_value = {"assessment_id": "A-1"}
        execute_tool = make_execute_tool(mock_conn)

        execute_tool(
            "persist_assessment",
            {"entity_id": "LOAN-0001", "entity_type": "LoanApplication",
             "regulation_id": "APS-112", "verdict": "COMPLIANT",
             "confidence": 0.9, "findings": [], "reasoning_steps": []},
        )

        mock_fn.assert_called_once()

    @patch("src.agent.dispatcher.trace_evidence")
    def test_routes_trace_evidence(self, mock_fn, mock_conn):
        mock_fn.return_value = {"assessment": {}}
        execute_tool = make_execute_tool(mock_conn)

        execute_tool("trace_evidence", {"assessment_id": "ASSESS-001"})

        mock_fn.assert_called_once_with(assessment_id="ASSESS-001", conn=mock_conn)

    @patch("src.agent.dispatcher.evaluate_thresholds")
    def test_routes_evaluate_thresholds(self, mock_fn, mock_conn):
        mock_fn.return_value = {"results": []}
        execute_tool = make_execute_tool(mock_conn)

        execute_tool(
            "evaluate_thresholds",
            {"entity_id": "LOAN-0001", "entity_type": "LoanApplication",
             "thresholds": []},
        )

        mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------


class TestUnknownTool:
    def test_returns_error_for_unknown_tool(self, mock_conn):
        execute_tool = make_execute_tool(mock_conn)
        result = execute_tool("nonexistent_tool", {})

        assert "error" in result
        assert "Unknown tool" in result["error"]


# ---------------------------------------------------------------------------
# In-session caching
# ---------------------------------------------------------------------------


class TestToolCaching:
    @patch("src.agent.dispatcher.traverse_compliance_path")
    def test_cacheable_tool_returns_cached_result(self, mock_fn, mock_conn):
        mock_fn.return_value = {"entity": {}, "regulations": {}}
        execute_tool = make_execute_tool(mock_conn)

        inputs = {"entity_id": "LOAN-0001", "entity_type": "LoanApplication"}
        result1 = execute_tool("traverse_compliance_path", inputs)
        result2 = execute_tool("traverse_compliance_path", inputs)

        assert result1 == result2
        assert mock_fn.call_count == 1  # second call served from cache

    @patch("src.agent.dispatcher.traverse_compliance_path")
    def test_different_inputs_not_cached(self, mock_fn, mock_conn):
        mock_fn.return_value = {"entity": {}, "regulations": {}}
        execute_tool = make_execute_tool(mock_conn)

        execute_tool(
            "traverse_compliance_path",
            {"entity_id": "LOAN-0001", "entity_type": "LoanApplication"},
        )
        execute_tool(
            "traverse_compliance_path",
            {"entity_id": "LOAN-0002", "entity_type": "LoanApplication"},
        )

        assert mock_fn.call_count == 2

    @patch("src.agent.dispatcher.detect_graph_anomalies")
    def test_non_cacheable_tool_not_cached(self, mock_fn, mock_conn):
        mock_fn.return_value = {"patterns_run": 0, "total_findings": 0, "results": []}
        execute_tool = make_execute_tool(mock_conn)

        inputs = {"pattern_names": ["high_lvr_loans"]}
        execute_tool("detect_graph_anomalies", inputs)
        execute_tool("detect_graph_anomalies", inputs)

        assert mock_fn.call_count == 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_tool_exception_returns_error_dict(self, mock_conn):
        mock_conn.run_query.side_effect = Exception("Connection lost")
        execute_tool = make_execute_tool(mock_conn)

        result = execute_tool(
            "read-neo4j-cypher",
            {"query": "MATCH (n) RETURN n LIMIT 1"},
        )

        assert "error" in result
        assert "Connection lost" in result["error"]
