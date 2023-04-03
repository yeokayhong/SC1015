# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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

x = bike_data.loc[:, ["Cooling system_Air", "Cooling system_Liquid", "Cooling system_Oil & air", "Torque (Nm)", "Displacement (ccm)"]]
y = bike_data["Power (hp)"]

# %%
