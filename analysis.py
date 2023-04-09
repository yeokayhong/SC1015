# %%
heart_data.info()
heart_data.describe()

# %%
def autopct_format(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{:.1f}%\n({v:d})'.format(pct, v=val)
    return my_format

colours = {0: 'C0',
           1: 'C1'}

for column in heart_data.columns:
    if column == "HeartDisease":
        continue
    if (heart_data[column].dtype not in ["object", "category", "bool"]):
        seaborn.violinplot(x = "HeartDisease", y = column, data = heart_data)
    else:
        print(column)
        # plot a pie chart for each category of the column based on whether they have stroke
        for category in heart_data[column].unique():
            data = heart_data[heart_data[column] == category]
            count = data["HeartDisease"].value_counts()
            plt.pie(count, labels=count.index, colors=[colours[key] for key in count.index], autopct=autopct_format(count))
            plt.title(category)
            plt.show()
        plt.show()
    # seaborn.catplot(x=columns, y="Score", data=grade_data, kind="violin")
    plt.show()

# %%
x = heart_data.loc[:, ["Sex", "ChestPainType", "Cholesterol", "FastingBS", "ExerciseAngina", "Oldpeak", "ST_Slope"]]
y = heart_data["HeartDisease"]