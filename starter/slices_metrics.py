# Import needed libraries
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from ml.model import compute_model_metrics, inference
from ml.data import process_data

# List of categorical features to slice
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

# Load data and saved models
data = pd.read_csv("data/census_mod.csv", sep=",")
model = joblib.load("model/lr_model.pkl")
encoder = joblib.load("model/lr_encoder.pkl")
lb = joblib.load("model/lr_lb.pkl")

# Split the data in train-test
_, test = train_test_split(data, test_size=0.20)

# Compute the metrics for the complete test dataset
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

# Print the results into the output txt file
with open("model/slice_output.txt", "w") as f:
    f.write("complete test dataset:\n")
    f.write("\tprecision: " + str(precision) + "\n")
    f.write("\trecall: " + str(recall) + "\n")
    f.write("\tfbeta: " + str(fbeta) + "\n")

# Itreate through each feature and value to compute the 'sliced' metrics
for cat in cat_features:
    with open("model/slice_output.txt", "a") as f:
        f.write("feature:" + cat + "\n")
    for val in test[cat].unique():
        test_slice = test[test[cat] == val]
        # print(val)
        X_test, y_test, _, _ = process_data(
            test_slice, categorical_features=cat_features, label="salary", training=False,
            encoder=encoder, lb=lb
        )
        preds = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        with open("model/slice_output.txt", "a") as f:
            f.write("\tfixed value: " + val + "\n")
            f.write("\t\tprecision: " + str(precision) + "\n")
            f.write("\t\trecall: " + str(recall) + "\n")
            f.write("\t\tfbeta: " + str(fbeta) + "\n")
