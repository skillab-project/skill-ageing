import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from pathlib import Path
from skill_api_without_cred import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def cleanup_cache():
    """Ensure cache is clean before and after tests."""
    folder = Path("Completed_Analyses")
    if folder.exists():
        for f in folder.glob("*.json"): f.unlink()
    yield
    if folder.exists():
        for f in folder.glob("*.json"): f.unlink()

def test_run_skill_analysis_logic():
    from skill_api_without_cred import run_skill_analysis_from_list
    mock_data = [
        {"upload_date": "2023-01-01", "skills": ["Python"]},
        {"upload_date": "2023-02-01", "skills": ["Python"]}
    ]
    result = run_skill_analysis_from_list(mock_data)
    assert result["summary"]["Total Skills Found"] == 1
    assert result["data"]["total_jobs_analyzed"] == 2

@patch("requests.post")
def test_analyze_jobs_endpoint(mock_post):
    # Mock Token
    m_token = MagicMock()
    m_token.text = '"mock_token"'
    m_token.status_code = 200

    # Mock Jobs API
    m_jobs = MagicMock()
    m_jobs.json.return_value = {
        "count": 1,
        "items": [{"upload_date": "2023-01-01", "skills": ["http://s1"]}]
    }
    m_jobs.status_code = 200

    # Mock Skill Resolver
    m_skills = MagicMock()
    m_skills.json.return_value = {
        "items": [{"id": "http://s1", "label": "python"}]
    }
    m_skills.status_code = 200

    # Mock Tracker Total
    m_total = MagicMock()
    m_total.json.return_value = {"count": 100}
    m_total.status_code = 200

    # Sequence: 1. Login, 2. Fetch Jobs, 3. Resolve Skills, 4. Login (for total), 5. Fetch Total
    mock_post.side_effect = [m_token, m_jobs, m_skills, m_token, m_total]

    response = client.get("/skill-ageing-jobs?occupation_ids=C2512")
    
    assert response.status_code == 200
    data = response.json()
    # If the endpoint ran correctly, we should have a biology summary
    assert "summary" in data
    # Ensure the processing lock was replaced by actual data
    assert data.get("status") != "processing"