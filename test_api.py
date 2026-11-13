
from fastapi.testclient import TestClient
from main import app, model

client = TestClient(app)


# Test 1: Home endpoint
def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]


# Test 2: Health endpoint
def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


# Test 3: Prediction endpoint
def test_predict():
    input_data = {
        "tenure": 12,
        "monthly_charges": 85.50,
        "total_charges": 1026.00
    }

    response = client.post("/predict", json=input_data)

    assert response.status_code == 200

    result = response.json()

    assert "churn_prediction" in result
    assert "churn_probability" in result
    assert "risk_level" in result

    assert result["churn_prediction"] in [0, 1]
    assert 0 <= result["churn_probability"] <= 100
    assert result["risk_level"] in ["High Risk", "Low Risk"]


# Test 4: Invalid input validation
def test_invalid_input():
    input_data = {
        "tenure": -5,
        "monthly_charges": 85.50,
        "total_charges": 1026.00
    }

    response = client.post("/predict", json=input_data)

    assert response.status_code == 422
