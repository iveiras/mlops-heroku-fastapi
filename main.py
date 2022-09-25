# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import os
import pandas as pd

from starter.ml.model import inference

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 52,
                "workclass": "Self-emp-not-inc",
                "fnlgt": 209642,
                "education": "HS-grad",
                "education_num": 9,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 45,
                "native_country": "United-States",
            }
        }


app = FastAPI()

@app.get("/")
async def welcome():
    return {"message": "This is the salary prediction API based on census data."}

@app.post("/predict")
async def predict_salary(record: CensusData):
    input_census = pd.read_json(record)
    salary = inference(input_census)
    return salary
