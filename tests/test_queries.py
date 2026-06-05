"""
Unit tests for src/graph/queries.py

Neo4j connection is fully mocked — no database required.
Covers Layer 1, Layer 2, and Layer 3 query helpers.
"""

import pytest
from unittest.mock import MagicMock, call

from src.graph.queries import (
    get_transactions_for_account,
    get_loans_by_risk,
    get_requirements_for_loan_type,
    get_compliance_path,
    get_entity_compliance_values,
    vector_search_chunks,
    get_assessments_for_entity,
    get_assessment_with_evidence,
    merge_assessment,
    merge_finding,
    merge_reasoning_step,
    batch_merge_findings,
    batch_merge_reasoning_steps,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_conn():
    return MagicMock()


# ===========================================================================
# LAYER 1 — Entity helpers
# ===========================================================================


class TestGetTransactionsForAccount:
    def test_returns_transactions(self, mock_conn):
        mock_conn.run_query.return_value = [
            {"transaction_id": "TXN-001", "amount": 5000},
        ]
        result = get_transactions_for_account(mock_conn, "ACC-0610")

        assert len(result) == 1
        assert result[0]["transaction_id"] == "TXN-001"

    def test_passes_account_id_and_limit(self, mock_conn):
        mock_conn.run_query.return_value = []
        get_transactions_for_account(mock_conn, "ACC-0001", limit=10)

        _, params = mock_conn.run_query.call_args[0]
        assert params["id"] == "ACC-0001"
        assert params["limit"] == 10

    def test_default_limit_is_50(self, mock_conn):
        mock_conn.run_query.return_value = []
        get_transactions_for_account(mock_conn, "ACC-0001")

        _, params = mock_conn.run_query.call_args[0]
        assert params["limit"] == 50


class TestGetLoansByRisk:
    def test_returns_loans(self, mock_conn):
        mock_conn.run_query.return_value = [
            {"loan_id": "LOAN-001", "risk_rating": "high"},
        ]
        result = get_loans_by_risk(mock_conn, "high")

        assert len(result) == 1
        assert result[0]["loan_id"] == "LOAN-001"

    def test_passes_risk_rating(self, mock_conn):
        mock_conn.run_query.return_value = []
        get_loans_by_risk(mock_conn, "medium", limit=25)

        _, params = mock_conn.run_query.call_args[0]
        assert params["risk"] == "medium"
        assert params["limit"] == 25


# ===========================================================================
# LAYER 2 — Regulatory helpers
# ===========================================================================


class TestGetRequirementsForLoanType:
    def test_with_regulation_filter(self, mock_conn):
        mock_conn.run_query.return_value = [{"requirement_id": "REQ-001"}]
        result = get_requirements_for_loan_type(
            mock_conn, "residential_secured", regulation_id="APG-223"
        )

        assert len(result) == 1
        _, params = mock_conn.run_query.call_args[0]
        assert params["reg_id"] == "APG-223"
        assert params["loan_type"] == "residential_secured"

    def test_without_regulation_filter(self, mock_conn):
        mock_conn.run_query.return_value = [{"requirement_id": "REQ-002"}]
        result = get_requirements_for_loan_type(mock_conn)

        assert len(result) == 1
        _, params = mock_conn.run_query.call_args[0]
        assert params["loan_type"] == "residential_secured"


class TestVectorSearchChunks:
    def test_with_regulation_filter(self, mock_conn):
        mock_conn.run_query.return_value = [{"chunk_id": "C-001", "score": 0.95}]
        result = vector_search_chunks(
            mock_conn, embedding=[0.1] * 1536, top_k=5, regulation_id="APS-112"
        )

        assert len(result) == 1
        _, params = mock_conn.run_query.call_args[0]
        assert params["reg_id"] == "APS-112"
        assert params["k"] == 5

    def test_without_regulation_filter(self, mock_conn):
        mock_conn.run_query.return_value = []
        vector_search_chunks(mock_conn, embedding=[0.1] * 10, top_k=3)

        _, params = mock_conn.run_query.call_args[0]
        assert "reg_id" not in params


# ===========================================================================
# get_compliance_path
# ===========================================================================


class TestGetCompliancePath:
    def test_loan_application_path(self, mock_conn):
        entity_rows = [
            {
                "loan_id": "LOAN-0002",
                "loan_type": "residential_secured",
                "amount": 500000,
                "lvr": 92.0,
                "interest_rate_pct": 6.5,
                "borrower_id": "BRW-0001",
                "borrower_name": "John",
                "risk_rating": "high",
                "jurisdiction_id": "JUR-AU-FED",
                "jurisdiction_name": "Australia",
                "jurisdiction_aml_risk": "low",
                "collateral_value": 550000,
            }
        ]
        reg_rows = [
            {
                "regulation_id": "APG-223",
                "regulation_name": "APG-223 Test",
                "is_enforceable": True,
                "section_id": "S1",
                "section_title": "Section 1",
                "requirement_id": "REQ-001",
                "requirement_description": "Test requirement",
                "severity": "HIGH",
                "is_quantitative": True,
                "threshold_id": "THR-001",
                "metric": "lvr",
                "operator": "<=",
                "threshold_value": 80,
                "unit": "%",
                "consequence": "Must not exceed",
                "threshold_type": "maximum",
            }
        ]
        mock_conn.run_query.side_effect = [entity_rows, reg_rows]

        result = get_compliance_path(
            mock_conn, entity_id="LOAN-0002", entity_type="LoanApplication"
        )

        assert result["entity"]["loan_id"] == "LOAN-0002"
        assert result["jurisdiction_id"] == "JUR-AU-FED"
        assert "APG-223" in result["regulations"]
        reg = result["regulations"]["APG-223"]
        assert reg["sections"]["S1"]["requirements"]["REQ-001"]["thresholds"][0]["metric"] == "lvr"

    def test_borrower_path(self, mock_conn):
        entity_rows = [
            {
                "borrower_id": "BRW-0001",
                "name": "Acme Corp",
                "risk_rating": "high",
                "jurisdiction_id": "JUR-AU-FED",
                "jurisdiction_aml_risk": "low",
            }
        ]
        mock_conn.run_query.side_effect = [entity_rows, []]

        result = get_compliance_path(
            mock_conn, entity_id="BRW-0001", entity_type="Borrower"
        )

        assert result["entity"]["borrower_id"] == "BRW-0001"
        assert result["jurisdiction_id"] == "JUR-AU-FED"

    def test_defaults_jurisdiction_when_missing(self, mock_conn):
        entity_rows = [{"borrower_id": "BRW-0001", "jurisdiction_id": None}]
        mock_conn.run_query.side_effect = [entity_rows, []]

        result = get_compliance_path(
            mock_conn, entity_id="BRW-0001", entity_type="Borrower"
        )

        assert result["jurisdiction_id"] == "JUR-AU-FED"

    def test_with_regulation_filter(self, mock_conn):
        mock_conn.run_query.side_effect = [
            [{"loan_id": "LOAN-001", "jurisdiction_id": "JUR-AU-FED", "loan_type": "residential_secured"}],
            [],
        ]

        get_compliance_path(
            mock_conn, entity_id="LOAN-001", entity_type="LoanApplication",
            regulation_id="APS-112",
        )

        # Second query should contain the regulation filter
        reg_cypher = mock_conn.run_query.call_args_list[1][0][0]
        reg_params = mock_conn.run_query.call_args_list[1][0][1]
        assert "reg_id" in reg_params
        assert reg_params["reg_id"] == "APS-112"

    def test_empty_entity_rows(self, mock_conn):
        mock_conn.run_query.side_effect = [[], []]

        result = get_compliance_path(
            mock_conn, entity_id="LOAN-9999", entity_type="LoanApplication"
        )

        assert result["entity"] == {}
        assert result["jurisdiction_id"] == "JUR-AU-FED"


# ===========================================================================
# get_entity_compliance_values
# ===========================================================================


class TestGetEntityComplianceValues:
    def test_loan_application(self, mock_conn):
        mock_conn.run_query.return_value = [
            {
                "lvr": 92.0,
                "interest_rate_pct": 6.5,
                "loan_amount": 500000,
                "serviceability_assessment_rate": 9.5,
                "income_type": "salary",
                "loan_type": "residential_secured",
                "non_salary_income_haircut_pct": None,
                "rental_income_gross": None,
                "rental_income_haircut_pct": None,
            }
        ]

        result = get_entity_compliance_values(
            mock_conn, entity_id="LOAN-0002", entity_type="LoanApplication"
        )

        assert result["lvr"] == 92.0
        assert result["serviceability_buffer_applied"] == 3.0
        assert "non_salary_income_haircut_pct" not in result  # None values excluded

    def test_borrower(self, mock_conn):
        mock_conn.run_query.return_value = [
            {"lvr": 85.0, "interest_rate_pct": 6.0, "serviceability_assessment_rate": None}
        ]

        result = get_entity_compliance_values(
            mock_conn, entity_id="BRW-0001", entity_type="Borrower"
        )

        assert result["lvr"] == 85.0
        # No derived buffer since serviceability_assessment_rate is None
        assert "serviceability_buffer_applied" not in result

    def test_empty_result(self, mock_conn):
        mock_conn.run_query.return_value = []

        result = get_entity_compliance_values(
            mock_conn, entity_id="LOAN-9999", entity_type="LoanApplication"
        )

        assert result == {}


# ===========================================================================
# LAYER 3 — Assessment helpers
# ===========================================================================


class TestGetAssessmentsForEntity:
    def test_returns_assessments(self, mock_conn):
        mock_conn.run_query.return_value = [
            {"assessment_id": "ASSESS-001", "verdict": "COMPLIANT"}
        ]

        result = get_assessments_for_entity(mock_conn, "LOAN-0002")

        assert len(result) == 1
        assert result[0]["assessment_id"] == "ASSESS-001"


class TestGetAssessmentWithEvidence:
    def test_returns_structured_evidence(self, mock_conn):
        # Query 1 returns assessment + findings
        assess_rows = [
            {
                "assessment_id": "ASSESS-001",
                "entity_id": "LOAN-0002",
                "entity_type": "LoanApplication",
                "regulation_id": "APG-223",
                "verdict": "NON_COMPLIANT",
                "confidence": 0.85,
                "agent": "compliance_agent",
                "created_at": "2024-01-01T00:00:00Z",
                "finding_id": "FIND-001",
                "finding_type": "threshold_breach",
                "severity": "HIGH",
                "f_description": "LVR exceeds 80%",
                "pattern_name": "high_lvr_loans",
            }
        ]
        # Query 2 returns reasoning steps
        step_rows = [
            {
                "step_number": 1,
                "description": "Checked LVR",
                "cypher_used": "MATCH (l) RETURN l",
                "cited_section_ids": ["S1"],
                "cited_chunk_ids": ["C1"],
                "cited_chunk_scores": [{"chunk_id": "C1", "score": 0.9}],
            }
        ]
        mock_conn.run_query.side_effect = [assess_rows, step_rows]

        result = get_assessment_with_evidence(mock_conn, "ASSESS-001")

        assert result["assessment"]["assessment_id"] == "ASSESS-001"
        assert len(result["findings"]) == 1
        assert result["findings"][0]["finding_id"] == "FIND-001"
        assert len(result["reasoning_steps"]) == 1

    def test_handles_no_findings(self, mock_conn):
        assess_rows = [
            {
                "assessment_id": "ASSESS-002",
                "entity_id": "LOAN-0001",
                "entity_type": "LoanApplication",
                "regulation_id": "APS-112",
                "verdict": "COMPLIANT",
                "confidence": 0.95,
                "agent": "compliance_agent",
                "created_at": "2024-01-01",
                "finding_id": None,
                "finding_type": None,
                "severity": None,
                "f_description": None,
                "pattern_name": None,
            }
        ]
        mock_conn.run_query.side_effect = [assess_rows, []]

        result = get_assessment_with_evidence(mock_conn, "ASSESS-002")

        assert result["findings"] == []


# ===========================================================================
# Layer 3 — Write helpers
# ===========================================================================


class TestMergeAssessment:
    def test_calls_run_query(self, mock_conn):
        mock_conn.run_query.return_value = []
        merge_assessment(
            mock_conn,
            assessment_id="ASSESS-001",
            entity_id="LOAN-0002",
            entity_type="LoanApplication",
            regulation_id="APG-223",
            verdict="COMPLIANT",
            confidence=0.9,
            agent="compliance_agent",
            created_at="2024-01-01T00:00:00Z",
        )

        mock_conn.run_query.assert_called_once()
        cypher = mock_conn.run_query.call_args[0][0]
        assert "MERGE" in cypher
        assert "Assessment" in cypher


class TestMergeFinding:
    def test_calls_run_query(self, mock_conn):
        mock_conn.run_query.return_value = []
        merge_finding(
            mock_conn,
            finding_id="FIND-001",
            assessment_id="ASSESS-001",
            finding_type="threshold_breach",
            severity="HIGH",
            description="LVR exceeds 80%",
            pattern_name="high_lvr_loans",
            related_entity_id=None,
            related_entity_type=None,
        )

        mock_conn.run_query.assert_called_once()


class TestMergeReasoningStep:
    def test_creates_step_and_citations(self, mock_conn):
        mock_conn.run_query.return_value = []
        merge_reasoning_step(
            mock_conn,
            step_id="STEP-001",
            assessment_id="ASSESS-001",
            step_number=1,
            description="Checked LVR thresholds",
            cypher_used="MATCH (l) RETURN l.lvr",
            section_ids=["S1", "S2"],
            chunk_ids=["C1"],
            chunk_scores={"C1": 0.92},
        )

        # 1 main step + 2 section citations + 1 chunk citation = 4 queries
        assert mock_conn.run_query.call_count == 4

    def test_no_citations(self, mock_conn):
        mock_conn.run_query.return_value = []
        merge_reasoning_step(
            mock_conn,
            step_id="STEP-002",
            assessment_id="ASSESS-001",
            step_number=2,
            description="Summary",
            cypher_used=None,
            section_ids=[],
            chunk_ids=[],
        )

        # Only the main step merge
        assert mock_conn.run_query.call_count == 1


class TestBatchMergeFindings:
    def test_batch_inserts_findings(self, mock_conn):
        mock_conn.run_query.return_value = []
        findings = [
            {"finding_id": "F1", "finding_type": "breach", "severity": "HIGH",
             "description": "test", "pattern_name": None},
            {"finding_id": "F2", "finding_type": "info", "severity": "LOW",
             "description": "test2", "pattern_name": None},
        ]

        batch_merge_findings(mock_conn, "ASSESS-001", findings)

        mock_conn.run_query.assert_called_once()
        params = mock_conn.run_query.call_args[0][1]
        assert params["aid"] == "ASSESS-001"
        assert len(params["rows"]) == 2

    def test_empty_findings_skips_query(self, mock_conn):
        batch_merge_findings(mock_conn, "ASSESS-001", [])
        mock_conn.run_query.assert_not_called()


class TestBatchMergeReasoningSteps:
    def test_batch_inserts_steps_and_citations(self, mock_conn):
        mock_conn.run_query.return_value = []
        steps = [
            {
                "step_id": "STEP-1",
                "step_number": 1,
                "description": "Check LVR",
                "cypher_used": "MATCH (l) RETURN l",
                "section_ids": ["S1"],
                "chunk_ids": ["C1"],
                "chunk_scores": {"C1": 0.9},
            },
        ]

        batch_merge_reasoning_steps(mock_conn, "ASSESS-001", steps)

        # 1 step merge + 1 section batch + 1 chunk batch = 3 queries
        assert mock_conn.run_query.call_count == 3

    def test_no_citations(self, mock_conn):
        mock_conn.run_query.return_value = []
        steps = [
            {
                "step_id": "STEP-1",
                "step_number": 1,
                "description": "Summary",
                "cypher_used": None,
                "section_ids": [],
                "chunk_ids": [],
            },
        ]

        batch_merge_reasoning_steps(mock_conn, "ASSESS-001", steps)

        # Only the main step merge
        assert mock_conn.run_query.call_count == 1

    def test_empty_steps_skips_query(self, mock_conn):
        batch_merge_reasoning_steps(mock_conn, "ASSESS-001", [])
        mock_conn.run_query.assert_not_called()
