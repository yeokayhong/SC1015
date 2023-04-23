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

## Exploratory Analysis
- We performed univariate exploratory analysis on each model, utilizing violinplots for numerical variables, and countplots for categorical variables
- We then split each of these plots by the presence of heart disease to attempt to visually identify patterns
- We then used numerical methods such as correlation matrix and chi-squared test to quantitatively verify our observations

## Machine Learning Techniques
### Models Used
- Random Forest Classifier (from `sklearn.ensemble`)
- Support Vector Classifier (from `sklearn.linear_model`)
- Logistic Regression (from `sklearn.svm`)

### Optimizing Performance
- Before we passed the data to the models, we standardized the input features using `StandardScaler` to ensure all features have the same scale to hopefully improve the model.
- We used `GridSearchCV` to find the optimal set of parameters
  - Set the scoring to be `recall` instead of the default `accuracy` to maximize true positive rates
  - Searched for the optimal `class_weight` to use for each model to maximize the recall score

### Evaluation of Models
- Printed each score metric
- Plotted heatmap of confusion matrix for each model
- Visualized predicted probabilities of logistic regression with violinplot

## Contributions 
**Prakritipong Phuvajakrt**
- Random Forest Classifier Implementation
- Support Vector Classifier Implementation
 
**Yeo Kay Hong**
- Logistic Regression Implementation
- Exploratory Analysis
- README
 
**Low Zhan Rong, Jodian**
- Problem Research & Fact Checking
- Data Cleaning
- GridSearch Implementation

## References
- Principal causes of death. Ministry of Health. (2021). Retrieved April 12, 2023, from https://www.moh.gov.sg/resources-statistics/singapore-health-facts/principal-causes-of-death ​
- Mayo Foundation for Medical Education and Research. (2022, August 25). Heart disease. Mayo Clinic. Retrieved April 12, 2023, from https://www.mayoclinic.org/diseases-conditions/heart-disease/diagnosis-treatment/drc-20353124 ​
- Verma, Y. (2021, October 7). Why data scaling is important in machine learning & how to effectively do it. Analytics India Magazine. Retrieved April 12, 2023, from https://analyticsindiamag.com/why-data-scaling-is-important-in-machine-learning-how-to-effectively-do-it/ ​
- Raj, A. (2020, October 3). Unlocking the True Power of Support Vector Regression. https://towardsdatascience.com/unlocking-the-true-power-of-support-vector-regression-847fd123a4a0%E2%80%8B