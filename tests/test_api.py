import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_survival():
    payload = {
        "Pclass": 3,
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Sex_male": 1,
        "Embarked_Q": 0,
        "Embarked_S": 1,
        "FamilySize": 2,
        "IsAlone": 1,
        "Cabin_U": 1,
        "TicketPrefix_NoPrefix": 1,
        "Title_Miss": 0,
        "Title_Mrs": 0,
        "Title_Other": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "survived" in response.json()
