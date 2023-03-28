# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import pandas

# %%
grade_data = pandas.read_csv("data/data.csv")

grade_data["sex"] = grade_data["sex"].astype("category")
grade_data["studytime"] = grade_data["studytime"].astype("category")
grade_data["freetime"] = grade_data["freetime"].astype("category")
grade_data["romantic"] = grade_data["romantic"].astype("category")
grade_data["Walc"] = grade_data["Walc"].astype("category")
grade_data["goout"] = grade_data["goout"].astype("category")
grade_data["Parents_edu"] = grade_data["Parents_edu"].astype("category")
grade_data["reason"] = grade_data["reason"].astype("category")
grade_data["Grade"] = grade_data["G3"].apply(lambda x: 1 if x >= 15 else 2 if x >= 10 else 3 if x >= 5 else 4 if x >= 0 else "E")
grade_data = grade_data.drop(["G3", "Pass"], axis=1)

grade_data.info()
grade_data.describe()

# %%
for column in grade_data.columns:
    seaborn.boxplot(x = column, y = "Score", data = grade_data, order = grade_data.groupby(column)["Score"].median().sort_values().index)
    # seaborn.catplot(x=columns, y="Score", data=grade_data, kind="violin")
    plt.show()

# %%
grade_data = grade_data[grade_data["absences"] != 4]
grade_data = grade_data[grade_data["absences"] != 2]
grade_data = grade_data[grade_data["absences"] != 6]

grade_data = grade_data[grade_data["goout"] != "1"]

# %%
for column in grade_data.columns:
    seaborn.swarmplot(x = column, y = "Score", data = grade_data,
           order = grade_data.groupby(column)["Score"].median().sort_values().index)
    plt.show()

for column in grade_data.columns:
    seaborn.boxplot(x = column, y = "Score", data = grade_data,
        order = grade_data.groupby(column)["Score"].median().sort_values().index)
    plt.show()
# %%

grade_data.drop(["romantic", "absences", "reason"], axis=1, inplace=True)

# %%
grade_data["sex"] = grade_data["sex"].apply(lambda x: 1 if x == "M" else 0)
grade_data["Walc"] = grade_data["Walc"].astype("int")
grade_data["Parents_edu"] = grade_data["Parents_edu"].astype("int")
grade_data["studytime"] = grade_data["studytime"].astype("int")
grade_data["freetime"] = grade_data["freetime"].astype("int")
grade_data["goout"] = grade_data["goout"].astype("int")

x = grade_data.drop("Grade", axis=1)
y = grade_data["Grade"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# %%
# use random forest regressor to predict the score
# Import GridSearch for hyperparameter tuning using Cross-Validation (CV)
from sklearn.model_selection import GridSearchCV

# Define the Hyper-parameter Grid to search on, in case of Random Forest
param_grid = {'n_estimators': np.arange(100,1001,100),   # number of trees 100, 200, ..., 1000
              'max_depth': np.arange(2, 11)}             # depth of trees 2, 3, 4, 5, ..., 10

# Create the Hyper-parameter Grid
hpGrid = GridSearchCV(RandomForestClassifier(),   # the model family
                      param_grid,                 # the search grid
                      cv = 5,                     # 5-fold cross-validation
                      scoring = 'accuracy')       # score to evaluate

# Train the models using Cross-Validation
hpGrid.fit(x_train, y_train.ravel())

# Fetch the best Model or the best set of Hyper-parameters
print(hpGrid.best_estimator_)

# Print the score (accuracy) of the best Model after CV
print(np.abs(hpGrid.best_score_))

# %%

classifier = RandomForestClassifier(n_estimators = 600, max_depth = 4)
classifier.fit(x_train, y_train.ravel())
y_train_pred = classifier.predict(x_train)

print("Train Data")
print("Accuracy  :\t", classifier.score(x_train, y_train))
print()

cmTrain = confusion_matrix(y_train, y_train_pred)
tpTrain = cmTrain[1][1] # True Positives : Good (1) predicted Good (1)
fpTrain = cmTrain[0][1] # False Positives : Bad (0) predicted Good (1)
tnTrain = cmTrain[0][0] # True Negatives : Bad (0) predicted Bad (0)
fnTrain = cmTrain[1][0] # False Negatives : Good (1) predicted Bad (0)

print("TPR Train :\t", (tpTrain/(tpTrain + fnTrain)))
print("TNR Train :\t", (tnTrain/(tnTrain + fpTrain)))
print()

print("FPR Train :\t", (fpTrain/(tnTrain + fpTrain)))
print("FNR Train :\t", (fnTrain/(tpTrain + fnTrain)))

seaborn.heatmap(confusion_matrix(y_train, y_train_pred), 
           annot = True, fmt=".0f", annot_kws={"size": 18})

# %%
y_test_pred = classifier.predict(x_test)

print("Train Data")
print("Accuracy  :\t", classifier.score(x_test, y_test))
print()

cmTrain = confusion_matrix(y_test, y_test_pred)
tpTrain = cmTrain[1][1] # True Positives : Good (1) predicted Good (1)
fpTrain = cmTrain[0][1] # False Positives : Bad (0) predicted Good (1)
tnTrain = cmTrain[0][0] # True Negatives : Bad (0) predicted Bad (0)
fnTrain = cmTrain[1][0] # False Negatives : Good (1) predicted Bad (0)

print("TPR Train :\t", (tpTrain/(tpTrain + fnTrain)))
print("TNR Train :\t", (tnTrain/(tnTrain + fpTrain)))
print()

print("FPR Train :\t", (fpTrain/(tnTrain + fpTrain)))
print("FNR Train :\t", (fnTrain/(tpTrain + fnTrain)))

seaborn.heatmap(confusion_matrix(y_test, y_test_pred), 
           annot = True, fmt=".0f", annot_kws={"size": 18})
# %%
