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
bike_data = bike_data.drop()

bike_data.info()
bike_data.describe()

# %%
# create a grid of plots
fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(10, 10))
for column in bike_data.columns:
    try:
        if bike_data[column].dtype == "int64" or bike_data[column].dtype == "float64":
            # seaborn.violinplot(data = bike_data[column])
            seaborn.violinplot(data = bike_data[column], ax=axes.flatten()[bike_data.columns.get_loc(column)], title=column)
        else:
            # seaborn.histplot(data = bike_data[column])
            seaborn.histplot(data = bike_data[column], ax=axes.flatten()[bike_data.columns.get_loc(column)], title=column)
        plt.show()
    except:
        print(f"{bike_data[column]} {bike_data[column].dtype}")
# plt.show()

