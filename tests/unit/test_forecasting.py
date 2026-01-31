import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from skill_api_without_cred import app # Import app because it includes the router

client = TestClient(app)

@patch("requests.get")
def test_ku_forecast_arima(mock_get):
    """Test the KU Forecast endpoint."""
    
    # Generate 10 months of mock data so ARIMA has enough points
    mock_data = []
    for i in range(1, 11):
        month = f"2023-{i:02d}-01T00:00:00"
        mock_data.append({
            "timestamp": month,
            "detected_kus": {"K1": "1", "K2": "1"},
            "organization": "TestOrg"
        })

    mock_res = MagicMock()
    mock_res.json.return_value = mock_data
    mock_res.status_code = 200
    mock_get.return_value = mock_res

    response = client.get("/forecast/ku_forecast_arima?horizon=3")
    
    assert response.status_code == 200
    json_data = response.json()
    assert "results" in json_data
    assert "K1" in json_data["results"]
    
    # Check if prediction exists
    prediction = json_data["results"]["K1"]["prediction"]
    assert len(prediction) == 3
    assert "absolute" in prediction[0]

@patch("requests.post")
def test_policy_skill_forecast_error_handling(mock_post):
    """Test how the forecast handles empty API responses."""
    
    # Mock Login
    mock_login = MagicMock()
    mock_login.text = '"token"'
    
    # Mock empty policy response
    mock_policies = MagicMock()
    mock_policies.json.return_value = {"items": []}
    mock_policies.status_code = 200
    
    mock_post.side_effect = [mock_login, mock_policies]

    response = client.get("/forecast/policy_skill_forecast?keywords=ai")
    assert response.status_code == 200
    assert "error" in response.json()