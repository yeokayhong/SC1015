# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import numpy as np
import seaborn
import pandas

# %%
bike_data = pandas.read_csv("data/all_bikez_curated.csv")

bike_data = bike_data[bike_data["Power (hp)"].notnull()]
bike_data["Stroke (mm)"] = bike_data["Stroke (mm)"].apply(lambda x: x.replace(",", "") if type(x) != float else x).astype("float64")
bike_data = bike_data.drop(["Model", "Fuel system", "Front brakes", "Rear brakes", "Front tire", "Rear tire", "Front suspension", "Rear suspension", "Color options"], axis=1)
bike_data = bike_data[bike_data["Torque (Nm)"].notnull()]
bike_data = bike_data[bike_data["Displacement (ccm)"].notnull()]
bike_data = bike_data[bike_data["Cooling system"].notnull()]

bike_data.info()
bike_data.describe()

# %%
# create a grid of plots
fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(10, 25))
plt.tight_layout()
index = 0
for column in bike_data.columns:
    try:
        row = index // 1
        col = index % 1
        if bike_data[column].dtype == "object":
            seaborn.boxplot(data = bike_data, x=column, y="Power (hp)", ax=axes[row], order = bike_data.groupby(column)["Power (hp)"].median().sort_values().index)
            index += 1
        # plt.show()
    except Exception as e:
        print(e)
        print(f"{column} {bike_data[column].dtype}")
plt.show()

# %%
seaborn.pairplot(bike_data, x_vars="Power (hp)")
plt.show()

print(bike_data.corr()["Power (hp)"].sort_values(ascending=False))

# %%
# one hot encode Cooling System
bike_data = pandas.get_dummies(bike_data, columns=["Cooling system"], prefix=["Cooling system"])
bike_data = bike_data.reset_index(drop=True)

x = bike_data.loc[:, ["Torque (Nm)", "Displacement (ccm)"]]
y = bike_data["Power (hp)"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %%
model = LinearRegression()
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)

# print the accuracy
print("Train Data")
print("Accuracy  :\t", model.score(x_train, y_train))
print("MSE       :\t", mean_squared_error(y_train, y_train_pred))
print()

# plot the predicted values against the actual values
plt.scatter(y_train, y_train_pred)
plt.xlabel("Actual Power (hp)")
plt.ylabel("Predicted Power (hp)")
plt.show()

# %%
y_test_pred = model.predict(x_test)

# print the accuracy
print("Test Data")
print("Accuracy  :\t", model.score(x_test, y_test))
print("MSE       :\t", mean_squared_error(y_test, y_test_pred))
print()

# plot the predicted values against the actual values
plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual Power (hp)")
plt.ylabel("Predicted Power (hp)")
plt.show()

# %%
sc_x = StandardScaler()
sc_y = StandardScaler()
scaled_x_train = sc_x.fit_transform(x_train)
scaled_y_train = sc_y.fit_transform(y_train.values.reshape(-1, 1))
regressor = SVR(kernel = 'rbf')
regressor.fit(scaled_x_train, scaled_y_train)

y_train_pred = sc_y.inverse_transform(regressor.predict(scaled_x_train).reshape(-1,1))

# plt predicted versus actual y
plt.scatter(y_train, y_train_pred)
plt.xlabel("Actual Power (hp)")
plt.ylabel("Predicted Power (hp)")
plt.show()

# print the accuracy
print("Train Data")
print("Accuracy  :\t", regressor.score(scaled_x_train, scaled_y_train))
print("MSE       :\t", mean_squared_error(y_train, y_train_pred))
print()

# %%
scaled_x_test = sc_x.fit_transform(x_test)
scaled_y_test = sc_y.fit_transform(y_test.values.reshape(-1, 1))

y_test_pred = sc_y.inverse_transform(regressor.predict(scaled_x_test).reshape(-1,1))

# plt predicted versus actual y
plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual Power (hp)")
plt.ylabel("Predicted Power (hp)")
plt.show()

# print the accuracy
print("Test Data")
print("Accuracy  :\t", regressor.score(scaled_x_test, scaled_y_test))
print("MSE       :\t", mean_squared_error(y_test, y_test_pred))
print()

# %%
