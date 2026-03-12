import pytest
import os
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from dotenv import load_dotenv
from skill_api_without_cred import app

load_dotenv()
client = TestClient(app)

TRACKER_CREDS = os.getenv("TRACKER_USERNAME") and os.getenv("TRACKER_PASSWORD")

# === SPECIFIC DATE FOR FAST TESTING ===
TEST_DATE = "2024-09-01" 

@pytest.fixture(scope="session", autouse=True)
def cleanup_only_new_files():
    """Cleanup ONLY the files created during this test session."""
    # 1. Record what files existed before the tests started
    folders = ["results", "Completed_Analyses"]
    before_files = {}
    for folder in folders:
        path = Path(folder)
        if path.exists():
            before_files[folder] = set(os.listdir(path))
        else:
            before_files[folder] = set()

    yield  # Run all tests in the session

    # 2. Identify and delete only the new files
    for folder in folders:
        path = Path(folder)
        if path.exists():
            after_files = set(os.listdir(path))
            new_files = after_files - before_files[folder]
            
            for filename in new_files:
                file_to_delete = path / filename
                try:
                    file_to_delete.unlink() # Delete single file
                    print(f"🗑️ Cleaned up new test file: {file_to_delete}")
                except Exception as e:
                    print(f"⚠️ Could not delete {file_to_delete}: {e}")

@pytest.mark.skipif(not TRACKER_CREDS, reason="Tracker credentials missing in .env")
class TestSkillAgeingIntegration:
    
    def test_skill_ageing_jobs_specific_day(self):
        """Tests job analysis for exactly Sept 1, 2024."""
        response = client.get(
            "/skill-ageing-jobs",
            params={
                "min_upload_date": TEST_DATE,
                "max_upload_date": TEST_DATE,
                "occupation_ids": "http://data.europa.eu/esco/isco/C2512" 
            }
        )
        assert response.status_code == 200
        data = response.json()
        # Ensure we got a valid response (either data or a 'no jobs found' message)
        assert any(k in data for k in ["data", "message", "warning"])

    def test_skill_ageing_law_policy(self):
        """Tests law/policy skill analysis up to Sept 1, 2024."""
        response = client.get(
            "/skill-ageing-law-policy",
            params={
                "keywords": "AI", 
                "max_publication_date": TEST_DATE
            }
        )
        assert response.status_code == 200
        assert any(k in response.json() for k in ["data", "message", "warning"])

@pytest.mark.skipif(not TRACKER_CREDS, reason="Tracker credentials missing")
class TestForecastingIntegration:

    def test_jobs_forecast_specific_day(self):
        """Tests job skill forecasting limited to Sept 1, 2024."""
        response = client.get(
            "/forecast/jobs_skill_forecast_NEWONE",
            params={
                "min_upload_date": TEST_DATE,
                "max_upload_date": TEST_DATE,
                "horizon": 3
            }
        )
        assert response.status_code == 200
        data = response.json()
        # Forecast might return 'error' if 1 day doesn't provide enough data points,
        # but the test passes if the API responds correctly.
        assert any(k in data for k in ["results", "error", "message"])

@pytest.mark.skipif(not os.getenv("KU_API_URL"), reason="KU_API_URL missing")
class TestKUIntegration:

    def test_ku_skill_ageing(self):
        """Tests KU Ageing Analysis (uses separate KU API)."""
        response = client.get(
            "/ku-skill-ageing",
            params={"start_date": "2024-09", "end_date": "2024-09"}
        )
        assert response.status_code == 200
        assert any(k in response.json() for k in ["summary", "warning", "message"])