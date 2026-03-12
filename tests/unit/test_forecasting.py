import pytest
import os
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from skill_api_without_cred import app, run_skill_analysis_from_list

client = TestClient(app)

@pytest.fixture
def file_tracker():
    """Tracks specific files to be deleted after the test."""
    files_to_delete = []
    yield files_to_delete
    
    for file_path in files_to_delete:
        p = Path(file_path)
        if p.exists():
            try:
                p.unlink()  # Deletes only this specific file
                print(f"\n🗑️ Deleted test file: {file_path}")
            except Exception as e:
                print(f"\n⚠️ Could not delete {file_path}: {e}")



def test_run_skill_analysis_logic(file_tracker):
    """Test the core logic and register the random result file for deletion."""
    mock_data = [
        {"upload_date": "2023-01-01", "skills": ["Python"]},
        {"upload_date": "2023-02-01", "skills": ["Python"]}
    ]
    
    result = run_skill_analysis_from_list(mock_data)
    
    if "file_saved" in result:
        file_tracker.append(result["file_saved"])
    
    assert result["data"]["total_jobs_analyzed"] == 2
    assert result["summary"]["Total Skills Found"] == 1


@patch("requests.post")
def test_analyze_jobs_endpoint(mock_post, file_tracker):
    """Test the endpoint and cleanup both result and cache files."""
    
    # 1. Mock Login Token
    mock_login = MagicMock()
    mock_login.text = '"mock_token"'
    mock_login.status_code = 200
    
    # 2. Mock Jobs Response 
    mock_jobs = MagicMock()
    mock_jobs.json.return_value = {
        "count": 1,
        "items": [
            {
                "upload_date": "2024-01-01",
                "skills": ["http://data.europa.eu/esco/skill/s1"] 
            }
        ]
    }
    mock_jobs.status_code = 200

    # 3. Mock ESCO Skills mapping response
    mock_esco = MagicMock()
    mock_esco.json.return_value = {
        "items": [{"id": "http://data.europa.eu/esco/skill/s1", "label": "python"}]
    }
    mock_esco.status_code = 200

    # 4. Mock Tracker Total (used at the end of the endpoint logic)
    mock_total = MagicMock()
    mock_total.json.return_value = {"count": 100}
    mock_total.status_code = 200

    mock_post.side_effect = [mock_login, mock_jobs, mock_esco, mock_login, mock_total]

    # --- Execution ---
    response = client.get("/skill-ageing-jobs?occupation_ids=C2512")
    
    assert response.status_code == 200
    data = response.json()

    # --- File Tracking for Cleanup ---
    if "file_saved" in data:
        file_tracker.append(data["file_saved"])

    cache_file = "Completed_Analyses/completed_analysis_skill_ageing_C2512.json"
    file_tracker.append(cache_file)

    # --- Assertions ---
    assert "summary" in data
    assert data["summary"]["Jobs Retrieved"] == 1
    # Verify label resolution from s1 -> python
    assert data["data"]["skill_biology_summary"][0]["Skill"] == "python"