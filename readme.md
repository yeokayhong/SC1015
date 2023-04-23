# Heart Disease Prediction with ML
In this project, we aim to predict the presence of heart disease in patients based on existing medical indicators using machine learning models.

## Overview
The project code is divided into several sections:
- Loading and preparing the dataset
- Performing analysis on the data
- Training the models
- Evaluating the model on train data
- Testing the model on test data

## Our dataset
We are using the `Heart Failure Prediction Dataset` from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

From the dataset, we have 6 predictors and 1 response variable

### Predictors
- **Age**: self-explanatory, numerical
- **MaxHR**: maximum heart rate, numerical
- **OldPeak**: amount of depression in the ST section of the ECG, numerical
- **ST Slope**: slope of the peak exercise ST section of the ECG, categorical {Up, Down, Flat}
- **ExerciseAngina**: whether patient experiences exercised-induced angina, categorical {Y - Yes, N - No}
- **ChestPainType**: type of chest pain patient experienced, categorical {ASY - Asymptomatic, TA - Typical Angina, NAP - Non-Anginal Pain, ATA - Atypical Angina}

### Response Variable
- **HeartDisease**: whether patient was diagnosed with heart disease, categorical {1 - Yes, 0 - No}

## Techniques
**Models Used**
- Random Forest Classifier (from `sklearn.ensemble`)
- Support Vector Classifier (from `sklearn.linear_model`)
- Logistic Regression (from `sklearn.svm`)

**Optimizing Performance**
- Before we passed the data to the models, we standardized the input features using `StandardScaler` to ensure all features have the same scale to hopefully improve the model.
- We used `GridSearchCV` to find the optimal set of parameters
  - Set the scoring to be `recall` instead of the default `accuracy` to maximize true positive rates
  - Searched for the optimal `class_weight` to use for each model to maximize the recall score

**Model Evaluations**: We are primarily concerned with the true positive rates for our models, however, we also need to keep an eye out for overall accuracy and true negative rates. So for all models, we calculated the confusion matrix and plotted them on a heatmap.

For the Logistic Regression model, as it returns probabilities as well, we plotted a violinplot for the distribution of the predicted probabilities against the actual label to better visualize the output.