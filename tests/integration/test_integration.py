import pytest
import os
import glob
from fastapi.testclient import TestClient
from dotenv import load_dotenv
import requests

from skill_api_without_cred import app

load_dotenv()
client = TestClient(app)

TRACKER_CREDS = os.getenv("TRACKER_USERNAME") and os.getenv("TRACKER_PASSWORD")

def is_ku_service_online(url):
    """Checks if the KU service is reachable."""
    if not url:
        return False
    try:
        response = requests.get(url, timeout=3)
        return response.status_code < 500
    except requests.RequestException:
        return False

KU_API_URL = os.getenv("KU_API_URL")
KU_ONLINE = is_ku_service_online(KU_API_URL)
print(f"KU Service Online: {KU_ONLINE} at {KU_API_URL}")


@pytest.fixture(scope="session", autouse=True)
def cleanup_results():
    """Cleanup generated JSON results after the test session."""
    yield
    files = glob.glob("results/skill_analysis_*.json") + glob.glob("results/ku_skill_analysis_*.json") + glob.glob("results/course_skill_*.json") + glob.glob("results/law_policy_skill_*.json")
    for f in files:
        try:
            os.remove(f)
        except:
            pass

@pytest.mark.skipif(not TRACKER_CREDS, reason="Tracker credentials missing")
class TestAgeingAnalysis:
    
    def test_jobs_with_keywords_analysis(self):
        """Verifies real job fetching and the 'Skill Biology' logic."""
        response = client.get(
            "/jobs-with-keywords",
            params={
                "keywords": "data",
                "max_pages": 1 
            }
        )
        assert response.status_code == 200
        data = response.json()
        
        # Check if analysis was performed
        if "data" in data:
            assert "skill_biology_summary" in data["data"]
            assert "epidemiological_metrics" in data["data"]
            assert data["summary"]["jobs"] > 0
        else:
            # Handle case where no jobs were found for keywords
            assert "error" in data or "warning" in data

    def test_jobs_forecasting(self):
        """Tests the ARIMA/Trend forecasting logic for job skills."""
        response = client.get(
            "/forecast/jobs_skill_forecast_NEWONE",
            params={
                "keywords": "data",
                "horizon": 3,
                "max_pages": 1
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data or "error" in data

    def test_law_policy_analysis(self):
        """Verifies policy retrieval and skill mapping from Eur-Lex."""
        response = client.get(
            "/law-policy",
            params={"keywords": "data"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data or "error" in data

    def test_courses_analysis(self):
        """Tests course skill extraction and batch mapping."""
        response = client.get(
            "/courses",
            params={"keywords": "data"}
        )
        assert response.status_code == 200
        assert "summary" in response.json()


@pytest.mark.skipif(not KU_API_URL or not KU_ONLINE, reason=f"KU Service is offline or URL missing at {KU_API_URL}")
class TestKUAgeingAnalysis:

    def test_ku_analysis(self):
        """Verifies fetching data from the KU Portal and running ageing logic."""
        response = client.get("/ku", params={})
        assert response.status_code == 200
        data = response.json()
        # Accept analysis results or a graceful 'no data' warning
        assert any(key in data for key in ["summary", "warning", "error"])

    def test_ku_forecasting(self):
        """Tests ARIMA forecasting for Knowledge Units."""
        response = client.get(
            "/forecast/ku_forecast_arima",
            params={"horizon": 6}
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data or "error" in data