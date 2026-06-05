"""
Unit tests for src/agent/anomaly_detector.py

Neo4j connection is fully mocked — no database required.
"""

import pytest
from unittest.mock import MagicMock

from src.agent.anomaly_detector import AnomalyDetector, _extract_entity_ids
from src.mcp.schema import AnomalyFinding, ANOMALY_REGISTRY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_conn():
    return MagicMock()


@pytest.fixture
def detector(mock_conn):
    return AnomalyDetector(mock_conn)


# ---------------------------------------------------------------------------
# _extract_entity_ids helper
# ---------------------------------------------------------------------------


class TestExtractEntityIds:
    def test_extracts_ids_from_rows(self):
        rows = [
            {"borrower_id": "BRW-001", "name": "A"},
            {"borrower_id": "BRW-002", "name": "B"},
        ]
        ids = _extract_entity_ids(rows, "high_risk_industry")
        assert ids == ["BRW-001", "BRW-002"]

    def test_skips_none_values(self):
        rows = [
            {"borrower_id": "BRW-001"},
            {"borrower_id": None},
        ]
        ids = _extract_entity_ids(rows, "high_risk_industry")
        assert ids == ["BRW-001"]

    def test_unknown_pattern_returns_empty(self):
        rows = [{"x": 1}]
        ids = _extract_entity_ids(rows, "nonexistent")
        assert ids == []


# ---------------------------------------------------------------------------
# AnomalyDetector.run()
# ---------------------------------------------------------------------------


class TestAnomalyDetectorRun:
    def test_run_known_pattern_returns_finding(self, detector, mock_conn):
        mock_conn.run_query.return_value = [
            {"loan_id": "LOAN-0002", "lvr": 95.0}
        ]

        finding = detector.run("high_lvr_loans")

        assert isinstance(finding, AnomalyFinding)
        assert finding.pattern_name == "high_lvr_loans"
        assert finding.evidence == [{"loan_id": "LOAN-0002", "lvr": 95.0}]
        assert "LOAN-0002" in finding.entity_ids

    def test_run_unknown_pattern_raises(self, detector):
        with pytest.raises(ValueError, match="Unknown pattern"):
            detector.run("nonexistent_pattern")

    def test_run_returns_empty_on_no_results(self, detector, mock_conn):
        mock_conn.run_query.return_value = []

        finding = detector.run("high_lvr_loans")

        assert finding.evidence == []
        assert finding.entity_ids == []

    def test_run_handles_query_exception(self, detector, mock_conn):
        mock_conn.run_query.side_effect = Exception("Connection lost")

        finding = detector.run("high_lvr_loans")

        assert finding.evidence == []

    def test_run_with_entity_id_scopes_loan(self, detector, mock_conn):
        mock_conn.run_query.return_value = []

        detector.run("high_lvr_loans", entity_id="LOAN-0002")

        cypher_arg = mock_conn.run_query.call_args[0][0]
        params_arg = mock_conn.run_query.call_args[0][1]
        assert "{loan_id: $eid}" in cypher_arg
        assert params_arg["eid"] == "LOAN-0002"

    def test_run_with_entity_id_scopes_borrower_pattern(self, detector, mock_conn):
        mock_conn.run_query.return_value = []

        detector.run("high_risk_industry", entity_id="BRW-0001")

        cypher_arg = mock_conn.run_query.call_args[0][0]
        params_arg = mock_conn.run_query.call_args[0][1]
        assert "{borrower_id: $eid}" in cypher_arg
        assert params_arg["eid"] == "BRW-0001"


# ---------------------------------------------------------------------------
# AnomalyDetector.run_all()
# ---------------------------------------------------------------------------


class TestAnomalyDetectorRunAll:
    def test_returns_only_findings_with_evidence(self, detector, mock_conn):
        # Return evidence only for the first call, empty for rest
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return [{"loan_id": "LOAN-0001", "lvr": 92}]
            return []

        mock_conn.run_query.side_effect = side_effect

        results = detector.run_all()

        assert len(results) == 1
        assert results[0].evidence != []

    def test_returns_empty_when_no_findings(self, detector, mock_conn):
        mock_conn.run_query.return_value = []

        results = detector.run_all()

        assert results == []

    def test_sorts_by_severity(self, detector, mock_conn):
        # Return evidence for all patterns to test sorting
        mock_conn.run_query.return_value = [{"borrower_id": "BRW-001", "loan_id": "LOAN-001"}]

        results = detector.run_all()

        if len(results) > 1:
            severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            for i in range(len(results) - 1):
                s1 = severity_order.get(results[i].severity, 9)
                s2 = severity_order.get(results[i + 1].severity, 9)
                assert s1 <= s2


# ---------------------------------------------------------------------------
# AnomalyDetector.run_for_entity()
# ---------------------------------------------------------------------------


class TestAnomalyDetectorRunForEntity:
    def test_loan_entity_runs_correct_patterns(self, detector, mock_conn):
        mock_conn.run_query.return_value = []

        detector.run_for_entity("LOAN-0001", "LoanApplication")

        # Should run high_lvr_loans and transaction_structuring
        assert mock_conn.run_query.call_count == 2

    def test_borrower_entity_runs_correct_patterns(self, detector, mock_conn):
        mock_conn.run_query.return_value = []

        detector.run_for_entity("BRW-0001", "Borrower")

        # Should run: high_risk_industry, high_risk_jurisdiction,
        # layered_ownership, guarantor_concentration, transaction_structuring
        assert mock_conn.run_query.call_count == 5

    def test_filters_evidence_for_entity(self, detector, mock_conn):
        mock_conn.run_query.return_value = [
            {"borrower_id": "BRW-0001", "name": "Target"},
            {"borrower_id": "BRW-9999", "name": "Other"},
        ]

        findings = detector.run_for_entity("BRW-0001", "Borrower")

        # Findings should be filtered to only include BRW-0001 evidence
        for f in findings:
            if f.pattern_name != "transaction_structuring":
                for row in f.evidence:
                    has_entity = (
                        "BRW-0001" in str(row.get("borrower_id", ""))
                        or "BRW-0001" in str(row.get("loan_id", ""))
                        or "BRW-0001" in str(row.get("ultimate_owner_id", ""))
                    )
                    assert has_entity
