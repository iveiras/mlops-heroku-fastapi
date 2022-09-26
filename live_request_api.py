# Import needed libraries
import requests
from main import CensusData


# Define input data
body = CensusData(
    **{
        "age": 32,
        "workclass": "Private",
        "fnlgt": 644162,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 400,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "Portugal",
        }
    )

# Request to the API and print results
response = requests.post("https://iveiras-mlops-udacity.herokuapp.com/inference", data=body.json(by_alias=True))
print("Status code:", response.status_code)
print(response.json())
