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
            p.unlink()  # Deletes only this specific file
            print(f"🗑️ Deleted test file: {file_path}")



def test_run_skill_analysis_logic(file_tracker):
    """Test the core logic and register the random result file for deletion."""
    mock_data = [{"upload_date": "2023-01-01", "skills": ["Python"]}]
    
    result = run_skill_analysis_from_list(mock_data)
    
    file_tracker.append(result["file_saved"])
    
    assert result["data"]["total_jobs_analyzed"] == 1

@patch("requests.post")
def test_analyze_jobs_endpoint(mock_post, file_tracker):
    """Test the endpoint and cleanup both result and cache files."""
    
    # --- Mock Setup ---
    mock_login = MagicMock()
    mock_login.text = '"mock_token"'
    mock_login.status_code = 200
    
    mock_jobs = MagicMock()
    mock_jobs.json.return_value = {
        "count": 1,
        "items": [{"upload_date": "2024-01-01", "skills": ["http://s1"]}]
    }
    mock_jobs.status_code = 200

    mock_esco = MagicMock()
    mock_esco.json.return_value = {"items": [{"id": "http://s1", "label": "python"}]}
    mock_esco.status_code = 200

    mock_total = MagicMock()
    mock_total.json.return_value = {"count": 100}
    mock_total.status_code = 200

    mock_post.side_effect = [mock_login, mock_jobs, mock_esco, mock_login, mock_total]

    # --- Execution ---
    response = client.get("/skill-ageing-jobs?occupation_ids=C2512")
    assert response.status_code == 200
    data = response.json()

    # --- File Tracking ---
    if "file_saved" in data:
        file_tracker.append(data["file_saved"])

    cache_file = "Completed_Analyses/completed_analysis_skill_ageing_C2512.json"
    file_tracker.append(cache_file)

    # --- Assertions ---
    assert data["summary"]["Jobs Retrieved"] == 1