import pandas
import os

# combine all files in data/original/test and data/original/train directory into a single csv
def combine_files():
    # get all files in data/original/test directory
    files = os.listdir('data/original/test')
    # create a list to store all the dataframes
    dfs = []
    # loop through all the files
    for file in files:
        # read the csv file into a dataframe
        df = pandas.read_csv('data/original/test/' + file)
        # append the dataframe to the list
        dfs.append(df)
    # concatenate all the dataframes into a single dataframe
    df = pandas.concat(dfs)
    # write the dataframe to a csv file
    df.to_csv('data/original/test.csv', index=False)

    # get all files in data/original/train directory
    files = os.listdir('data/original/train')
    # create a list to store all the dataframes
    dfs = []
    # loop through all the files
    for file in files:
        # read the csv file into a dataframe
        df = pandas.read_csv('data/original/train/' + file)
        # append the dataframe to the list
        dfs.append(df)
    # concatenate all the dataframes into a single dataframe
    df = pandas.concat(dfs)

    # write the dataframe to a csv file
    df.to_csv('data/combined/data.csv', index=False)

if __name__ == '__main__':
    combine_files()