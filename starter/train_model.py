# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model

# Add code to load in the data.
data = pd.read_csv("data/census_mod.csv", sep=",")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# Process the train data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary"
)

# Train and save a model.
model = train_model(X_train, y_train)
joblib.dump(model, "model/lr_model.pkl")
joblib.dump(encoder, "model/lr_encoder.pkl")
joblib.dump(lb, "model/lr_lb.pkl")