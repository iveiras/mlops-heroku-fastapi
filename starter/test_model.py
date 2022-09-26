# Import needed libraries
import pandas as pd
import sklearn
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference


# Load the data and split it in train-test
data = pd.read_csv("data/census_mod.csv", sep=",")
train, test = sklearn.model_selection.train_test_split(data, test_size=0.20)

# List categorical features
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

# Process train data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary"
)


# Test the process_data function
def test_process_data():
    _, _, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary"
    )
    assert type(encoder) == sklearn.preprocessing._encoders.OneHotEncoder
    assert type(lb) == sklearn.preprocessing._label.LabelBinarizer


# Test the train_model function
def test_train_model():
    model = train_model(X_train, y_train)
    assert type(model) == sklearn.linear_model._logistic.LogisticRegression


# Test the inference function
def test_inference():
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    assert len(X_train) == len(preds)


# Test the compute_model_metrics function
def test_compute_model_metrics():
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    precision, recall, fbeta = compute_model_metrics(y_train, preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
