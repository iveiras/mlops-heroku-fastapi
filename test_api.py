from fastapi.testclient import TestClient
from main import app
from main import CensusData

# Instantiate the testing client with our app.
client = TestClient(app)


# Test root GET method.
def test_get():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "This is the salary prediction API based on census data."


# Test POST inference method with <=50k result.
def test_salary_below():
    body = CensusData(
        **{
            "age": 19,
            "workclass": "Private",
            "fnlgt": 544091,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Married-AF-spouse",
            "occupation": "Adm-clerical",
            "relationship": "Wife",
            "race": "White",
            "sex": "Female",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 25,
            "native-country": "United-States",
        }
    )

    response = client.post("/inference", data=body.json(by_alias=True))
    assert response.status_code == 200
    assert response.json()["salary"] == "<=50K"


# Test POST inference method with >50k result.
def test_salary_above():
    body = CensusData(
        **{
            "age": 42,
            "workclass": "Private",
            "fnlgt": 159449,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 5178,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        }
    )

    response = client.post("/inference", data=body.json(by_alias=True))
    assert response.status_code == 200
    assert response.json()["salary"] == ">50K"
