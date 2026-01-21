import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from skill_api_without_cred import app, run_skill_analysis_from_list

client = TestClient(app)

# Sample data for logic testing
MOCK_JOB_LIST = [
    {"upload_date": "2023-01-01", "skills": ["Python", "Data Science"]},
    {"upload_date": "2023-02-01", "skills": ["Python", "Machine Learning"]},
    {"upload_date": "2023-03-01", "skills": ["Python"]},
]

def test_run_skill_analysis_logic():
    """Test the core data processing function directly."""
    result = run_skill_analysis_from_list(MOCK_JOB_LIST)
    
    # Check the top-level structure
    assert "data" in result
    analysis_data = result["data"]
    
    # Check the nested content
    assert "skill_biology_summary" in analysis_data
    assert "total_jobs_analyzed" in analysis_data
    assert analysis_data["total_jobs_analyzed"] == 3
    
    # Check if Python (present in all) has a total count of 3
    # Note: Your code doesn't lowercase in run_skill_analysis_from_list, 
    # but it's good to be safe.
    python_entry = next(
        (item for item in analysis_data["skill_biology_summary"] if item["Skill"] == "Python"), 
        None
    )
    assert python_entry is not None, "Python skill not found in results"
    assert python_entry["Total Jobs"] == 3

@patch("requests.post")
def test_analyze_jobs_endpoint(mock_post):
    """Test the /jobs-with-keywords endpoint with mocked API responses."""
    
    # 1. Mock Login Token
    mock_login = MagicMock()
    mock_login.text = '"mock_token"'
    mock_login.status_code = 200
    
    # 2. Mock Jobs Response
    mock_jobs = MagicMock()
    mock_jobs.json.return_value = {
        "items": [
            {
                "upload_date": "2024-01-01", 
                "skills": [{"id": "http://data.europa.eu/esco/skill/s1"}]
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

    # Set the side effect for the 3 POST calls made in the endpoint
    mock_post.side_effect = [mock_login, mock_jobs, mock_esco]

    response = client.get("/jobs-with-keywords?keywords=python&max_pages=1")
    
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert data["summary"]["jobs"] == 1
    # Check that it mapped the ID to label
    assert data["data"]["skill_biology_summary"][0]["Skill"] == "python"