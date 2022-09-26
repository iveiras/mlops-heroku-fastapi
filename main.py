# Load libraries needed
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
import pandas as pd
import joblib

# Load developed functions
from starter.ml.model import inference
from starter.ml.data import process_data

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


# Define alias_generator function
def hyphen_underscore(string: str) -> str:
    return string.replace('-', '_')


# Define main body class
class CensusData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13, alias="education-num")
    marital_status: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=2174, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


    class Config:
        alias_generator = hyphen_underscore


app = FastAPI()


# Define GET method
@app.get("/")
async def welcome():
    return {"message": "This is the salary prediction API based on census data."}


# Load saved models and categories
model = joblib.load("model/lr_model.pkl")
encoder = joblib.load("model/lr_encoder.pkl")
lb = joblib.load("model/lr_lb.pkl")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Define POST method
@app.post("/inference")
async def predict_salary(record: CensusData):
    # input_census = pd.read_json(record)
    input_census = pd.DataFrame.from_dict([record.dict(by_alias=True)])
    X_val, _, _, _ = process_data(input_census, categorical_features=cat_features, training=False, 
                                  encoder=encoder, lb=lb)
    preds = lb.inverse_transform(inference(model, X_val))
    return {"salary": str(preds[0])}
