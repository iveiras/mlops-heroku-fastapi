# Model Card

## Model Details
This model was trained by <a href="https://github.com/iveiras" target="_blank">iveiras</a> as a Udacity's MLOps Engineer nanodegree student in September 2022. A Logistic Regression model from sklearn was trained with the default hyperparameters using the <a href="https://archive.ics.uci.edu/ml/datasets/census+income" target="_blank">census income dataset from UCI</a>.

## Intended Use
The target of the model is to predict wheter a person's annual income would be below or above 50k$. This model is only intended to be used under academic purpouses and shouldn't be used as a real salary predictor.

## Training & Evaluation Data
The model was trained using the <a href="https://archive.ics.uci.edu/ml/datasets/census+income" target="_blank">census income dataset from UCI</a>, splitting it into 80%-20% train-test subsets.

## Metrics
The metrics used in this model are Precision, Recall, and F1. The performance of the model on the evalution dataset is the following:
* Precision = 0.730
* Recall = 0.263
* F1 = 0.387

## Ethical Considerations
The model training process didn't take into consideration any ethic-related analysis, but the results given by the model (if used) should be taken carefully. Considering that several demographic variables are included in the dataset (sex, race, etc.), the output predictions could be leveraged in an unethical way and could be detrimental to minority demographical groups.

## Caveats and Recommendations
A deeper analysis of the results should be done to enhance the model results. Trying different model types or doing an hyper-parameter grid search should be the first steps to get better results.