# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import seaborn
import pandas

# %%
heart_data = pandas.read_csv("data/heart.csv")

heart_data["ExerciseAngina"] = heart_data["ExerciseAngina"].apply(lambda x: 1 if x == "Y" else 0).astype("object")
heart_data["FastingBS"] = heart_data["FastingBS"].astype("object")