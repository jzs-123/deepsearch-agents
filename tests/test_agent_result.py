import pytest
from pydantic import ValidationError
from app.agent.result import AgentResult, StructuredResultError, retry_structure, structure_retry_budget


@pytest.mark.parametrize("payload", [
    {"answer": "missing status"},
    {"status": "success"},
    {"status": "success", "answer": 123},
    {"status": "unknown", "answer": "bad status"},
    {"status": "success", "answer": "   "},
    {"status": "success", "answer": "ok", "extra": True},
    {"status": "success", "answer": "ok", "sources": "fake"},
    {"status": "success", "answer": "ok", "artifacts": [{"path": 123}]},
])
def test_invalid_result_is_rejected(payload):
    with pytest.raises(ValidationError):
        AgentResult.model_validate(payload)


@pytest.mark.parametrize("status", ["success", "partial", "refused", "failed"])
def test_result_roundtrips_through_json(status):
    result = AgentResult(status=status, answer="answer")
    assert AgentResult.model_validate_json(result.model_dump_json()) == result
    assert result.schema_version == "2.0"


def test_markdown_wrapped_json_is_not_a_valid_result():
    with pytest.raises(ValidationError):
        AgentResult.model_validate_json('```json\n{"status":"success","answer":"x"}\n```')


def test_schema_retry_is_limited_to_two_corrections():
    token = structure_retry_budget.set([2])
    try:
        assert "AgentResult" in retry_structure(ValueError("missing answer"))
        assert "AgentResult" in retry_structure(ValueError("invalid type"))
        with pytest.raises(StructuredResultError):
            retry_structure(ValueError("still invalid"))
    finally:
        structure_retry_budget.reset(token)
