import unittest
from fastapi.testclient import TestClient

# IMPORTANT: import app correctly
from skill_api_without_cred import app, run_skill_analysis_from_list


class TestSkillAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create client ONCE for all tests"""
        cls.client = TestClient(app)

    # ---------------------------
    # BASIC ROUTE TESTS
    # ---------------------------

    def test_root_endpoint(self):
        """Root may or may not exist"""
        response = self.client.get("/")
        self.assertIn(response.status_code, [200, 404])

    def test_skill_ageing_requires_params(self):
        """Missing params should return validation error"""
        response = self.client.get("/skill-ageing")
        self.assertEqual(response.status_code, 422)

    def test_law_policy_endpoint_exists(self):
        response = self.client.get("/skill-ageing-law-policy")
        self.assertIn(response.status_code, [200, 422])

    def test_ku_skill_ageing_endpoint_exists(self):
        response = self.client.get("/ku-skill-ageing")
        self.assertIn(response.status_code, [200, 422])

    def test_course_skill_ageing_endpoint_exists(self):
        response = self.client.get("/skill-ageing-courses")
        self.assertIn(response.status_code, [200, 422])

    # ---------------------------
    # CORE LOGIC TEST (NO API)
    # ---------------------------

    def test_run_skill_analysis_from_list(self):
        """Test core analysis logic directly"""
        dummy_jobs = [
            {"upload_date": "2023-01-01", "skills": ["python", "sql"]},
            {"upload_date": "2023-02-01", "skills": ["python"]},
            {"upload_date": "2024-01-01", "skills": ["sql"]},
        ]

        result = run_skill_analysis_from_list(dummy_jobs)

        self.assertIsInstance(result, dict)
        self.assertIn("data", result)
        self.assertIn("skill_biology_summary", result["data"])
        self.assertGreater(len(result["data"]["skill_biology_summary"]), 0)


if __name__ == "__main__":
    unittest.main()
