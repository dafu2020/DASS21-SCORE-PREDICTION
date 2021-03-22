import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sqlalchemy import create_engine

pd.options.mode.chained_assignment = None
"""
score code.
"""


def import_data(PATH):
    """
    Import data to dataframes.
    :return: a dataframe containing dataset read from a csv file
    """

    CSV_DATA = "DASS21.csv"

    dataset = pd.read_csv(PATH + CSV_DATA,
                          skiprows=1,
                          encoding="ISO-8859-1",
                          sep=',',
                          names=('DASS_21', 'Openness', 'Restraint', 'Transcendence', 'Interpersonal',
                                 'GHQ_12', 'SEC', 'Age', 'Gender', 'Work', 'Student', 'Day', 'Sons',
                                 'Appreciation_of_beauty', 'Bravery', 'Creativity', 'Curiosity',
                                 'Fairness', 'Forgiveness', 'Gratitude', 'Honesty', 'Hope', 'Humilty',
                                 'Humor', 'Judgment', 'Kindness', 'Leadership', 'Love', 'Love_of_learning',
                                 'Perseverance', 'Perspective', 'Prudence', 'Self_regulation', 'Social_intelligence',
                                 'Spirituality', 'Teamwork', 'Zest'))
    # Show all columns.
    pd.set_option('display.max_columns', None)

    # Increase number of columns that display on one line.
    pd.set_option('display.width', 1000)

    return dataset


def model_visual(PATH):
    """
    Display distribution graphs for the dataset by groups.
    :param PATH: path
    """
    df = import_data(PATH)
    df["DASS_21 Group"] = pd.cut(df["DASS_21"], [0, 21, 42, 63], precision=0, labels=["0-20", "21-41", "42-62"])

    print("Low DASS_21 Score: (0-20)")
    DASS_group_1 = df[(df["DASS_21 Group"] == '0-20')]
    print(DASS_group_1.head())
    print(DASS_group_1.describe().transpose())

    # ============ DISPLAY DISTRIBUTION GRAPHS FOR SUBGROUPS
    # display_multi_graph(DASS_group_1)

    print("Medium DASS_21 Score: (21-41)")
    DASS_group_2 = df[(df["DASS_21 Group"] == '21-41')]
    print(DASS_group_2.head())
    print(DASS_group_2.describe().transpose())

    # display_multi_graph(DASS_group_2)

    print("High DASS_21 Score: (42-62)")
    DASS_group_3 = df[(df["DASS_21 Group"] == '42-62')]
    print(DASS_group_3.head())
    print(DASS_group_3.describe().transpose())

    display_multi_graph(DASS_group_3)


def display_multi_graph(df):
    """
    Display the distribution graphs for the dataset categorized by columns.
    :param df: dataframe
    """
    plt.subplots(nrows=7, ncols=6, figsize=(10, 5))

    plt.subplot(7, 6, 1)
    # Specifies total rows, columns and image #
    # where images are drawn clockwise.
    plt.hist(df["Openness"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Openness', fontsize=10)

    plt.subplot(7, 6, 2)
    plt.hist(df["Restraint"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Restraint', fontsize=10)

    plt.subplot(7, 6, 3)
    plt.hist(df["Transcendence"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Transcendence', fontsize=10)

    plt.subplot(7, 6, 4)
    plt.hist(df["Interpersonal"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Interpersonal', fontsize=10)

    plt.subplot(7, 6, 5)
    plt.hist(df["GHQ_12"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('GHQ_12', fontsize=10)

    plt.subplot(7, 6, 6)
    plt.hist(df["SEC"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('SEC', fontsize=10)

    plt.subplot(7, 6, 7)
    plt.hist(df["Age"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Age', fontsize=10)

    plt.subplot(7, 6, 8)
    plt.hist(df["Gender"], bins=5)
    t11 = ['Female', 'Male']
    plt.xticks(range(len(t11)), t11, rotation=50, fontsize=5)
    plt.title('Gender', fontsize=10)

    plt.subplot(7, 6, 9)
    plt.hist(df["Day"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Day', fontsize=10)

    plt.subplot(7, 6, 10)
    plt.hist(df["Sons"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Sons', fontsize=10)

    plt.subplot(7, 6, 11)
    plt.hist(df["Appreciation_of_beauty"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Appreciation_of_beauty', fontsize=10)

    plt.subplot(7, 6, 12)
    plt.hist(df["Bravery"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Bravery', fontsize=10)

    plt.subplot(7, 6, 13)
    plt.hist(df["Creativity"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Creativity', fontsize=10)

    plt.subplot(7, 6, 14)
    plt.hist(df["Curiosity"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Curiosity', fontsize=10)

    plt.subplot(7, 6, 15)
    plt.hist(df["Fairness"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Fairness', fontsize=10)

    plt.subplot(7, 6, 16)
    plt.hist(df["Forgiveness"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Forgiveness', fontsize=10)

    plt.subplot(7, 6, 17)
    plt.hist(df["Gratitude"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Gratitude', fontsize=10)

    plt.subplot(7, 6, 18)
    plt.hist(df["Honesty"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Honesty', fontsize=10)

    plt.subplot(7, 6, 19)
    plt.hist(df["Hope"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Hope', fontsize=10)

    plt.subplot(7, 6, 20)
    plt.hist(df["Humilty"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Humilty', fontsize=10)

    plt.subplot(7, 6, 21)
    plt.hist(df["Humor"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Humor', fontsize=10)

    plt.subplot(7, 6, 22)
    plt.hist(df["Judgment"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Judgment', fontsize=10)

    plt.subplot(7, 6, 23)
    plt.hist(df["Kindness"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Kindness', fontsize=10)

    plt.subplot(7, 6, 24)
    plt.hist(df["Leadership"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Leadership', fontsize=10)

    plt.subplot(7, 6, 25)
    plt.hist(df["Love"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Love', fontsize=10)

    plt.subplot(7, 6, 26)
    plt.hist(df["Love_of_learning"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Love_of_learning', fontsize=10)

    plt.subplot(7, 6, 27)
    plt.hist(df["Perseverance"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Perseverance', fontsize=10)

    plt.subplot(7, 6, 28)
    plt.hist(df["Perspective"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Perspective', fontsize=10)

    plt.subplot(7, 6, 29)
    plt.hist(df["Prudence"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Prudence', fontsize=10)

    plt.subplot(7, 6, 30)
    plt.hist(df["Self_regulation"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Self_regulation', fontsize=10)

    plt.subplot(7, 6, 31)
    plt.hist(df["Social_intelligence"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Social_intelligence', fontsize=10)

    plt.subplot(7, 6, 32)
    plt.hist(df["Spirituality"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Spirituality', fontsize=10)

    plt.subplot(7, 6, 33)
    plt.hist(df["Teamwork"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Teamwork', fontsize=10)

    plt.subplot(7, 6, 34)
    plt.hist(df["Zest"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Zest', fontsize=10)

    plt.subplot(7, 6, 35)
    plt.hist(df["Social_intelligence"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Social_intelligence', fontsize=10)

    plt.subplot(7, 6, 36)
    plt.hist(df["Work"], bins=22)
    plt.xlabel("DASS_21 Score", fontsize=5)
    plt.ylabel("Frequency", fontsize=5)
    plt.title('Work', fontsize=10)

    plt.subplot(7, 6, 37)
    plt.hist(df["Student"], bins=5)
    t12 = ['Other', 'Student']
    plt.xticks(range(len(t12)), t12, rotation=50, fontsize=5)
    plt.title('Student', fontsize=10)

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


def splitIntoThreeCategories(df, colName, newColName):
    """
    Split the dataset into three categories.
    :param df: dataframe
    :param colName: str
    :param newColName: str
    :return: dataframe
    """
    third = df[colName].quantile(0.33)
    twoThird = df[colName].quantile(0.66)

    # Adds new column to end of existing data frame.
    df[newColName] = 2
    colNum = len(df.keys()) - 1

    for i in range(0, len(df)):
        currentVal = df.iloc[i][colName]
        if currentVal < third:
            # Add rating to cell
            df.iat[i, colNum] = 0
        elif currentVal < twoThird:
            df.iat[i, colNum] = 1
    return df


def showQueryResult(sql, df):
    """
    Show the query result.
    :param sql: str
    :param df: dataframe
    :return: query result
    """
    # This code creates an in-memory table called 'Inventory'.
    engine = create_engine('sqlite://', echo=False)
    connection = engine.connect()
    df.to_sql(name='DASS21', con=connection, if_exists='replace', index=False)

    # This code performs the query.
    queryResult = pd.read_sql(sql, connection)
    return queryResult


def print_DASS21_score_in_group(PATH):
    """
    Group bar graph of all categories under different DASS21 score groups.
    :param PATH: path
    """
    df = import_data(PATH)
    adjustedDf = splitIntoThreeCategories(df, 'DASS_21', 'DASS_21_cat')

    print(adjustedDf.head())

    SQL = "SELECT DASS_21_cat, AVG(Openness), AVG(Restraint), AVG(SEC), " \
          "AVG(Transcendence), AVG(Interpersonal), AVG(GHQ_12)," \
          "AVG(Age), AVG(Work), AVG(Day)," \
          "AVG(Sons), AVG(Appreciation_of_beauty), AVG(Bravery)," \
          "AVG(Creativity), AVG(Curiosity), AVG(Fairness)," \
          "AVG(Gratitude), AVG(Honesty), AVG(Hope)," \
          "AVG(Humilty), AVG(Humor), AVG(Judgment), " \
          "AVG(Kindness), AVG(Leadership), AVG(Love), " \
          "AVG(Love_of_learning), AVG(Perseverance), AVG(Perspective), " \
          "AVG(Prudence), AVG(Self_regulation), AVG(Social_intelligence), " \
          "AVG(Spirituality), AVG(Teamwork), AVG(Zest) " \
          "  FROM DASS21 GROUP BY  DASS_21_cat"
    results = showQueryResult(SQL, adjustedDf)
    print(results)

    df = pd.DataFrame([
        # First row.
        ['g0', 'AVG(Openness)', results.iloc[0]['AVG(Openness)']],
        ['g0', 'AVG(Restraint)', results.iloc[0]['AVG(Restraint)']],
        ['g0', 'AVG(Transcendence)', results.iloc[0]['AVG(Transcendence)']],
        ['g0', 'AVG(Interpersonal)', results.iloc[0]['AVG(Interpersonal)']],
        ['g0', 'AVG(GHQ_12)', results.iloc[0]['AVG(GHQ_12)']],
        ['g0', 'AVG(SEC)', results.iloc[0]['AVG(SEC)']],
        ['g0', 'AVG(Age)', results.iloc[0]['AVG(Age)']],
        ['g0', 'AVG(Work)', results.iloc[0]['AVG(Work)']],
        ['g0', 'AVG(Day)', results.iloc[0]['AVG(Day)']],
        ['g0', 'AVG(Sons)', results.iloc[0]['AVG(Sons)']],
        ['g0', 'AVG(Appreciation_of_beauty)', results.iloc[0]['AVG(Appreciation_of_beauty)']],
        ['g0', 'AVG(Bravery)', results.iloc[0]['AVG(Bravery)']],
        ['g0', 'AVG(Creativity)', results.iloc[0]['AVG(Creativity)']],
        ['g0', 'AVG(Curiosity)', results.iloc[0]['AVG(Curiosity)']],
        ['g0', 'AVG(Fairness)', results.iloc[0]['AVG(Fairness)']],
        ['g0', 'AVG(Gratitude)', results.iloc[0]['AVG(Gratitude)']],
        ['g0', 'AVG(Honesty)', results.iloc[0]['AVG(Honesty)']],
        ['g0', 'AVG(Hope)', results.iloc[0]['AVG(Hope)']],
        ['g0', 'AVG(Humilty)', results.iloc[0]['AVG(Humilty)']],
        ['g0', 'AVG(Humor)', results.iloc[0]['AVG(Humor)']],
        ['g0', 'AVG(Judgment)', results.iloc[0]['AVG(Judgment)']],
        ['g0', 'AVG(Kindness)', results.iloc[0]['AVG(Kindness)']],
        ['g0', 'AVG(Leadership)', results.iloc[0]['AVG(Leadership)']],
        ['g0', 'AVG(Love)', results.iloc[0]['AVG(Love)']],
        ['g0', 'AVG(Love_of_learning)', results.iloc[0]['AVG(Love_of_learning)']],
        ['g0', 'AVG(Perseverance)', results.iloc[0]['AVG(Perseverance)']],
        ['g0', 'AVG(Perspective)', results.iloc[0]['AVG(Perspective)']],
        ['g0', 'AVG(Prudence)', results.iloc[0]['AVG(Prudence)']],
        ['g0', 'AVG(Self_regulation)', results.iloc[0]['AVG(Self_regulation)']],
        ['g0', 'AVG(Social_intelligence)', results.iloc[0]['AVG(Social_intelligence)']],
        ['g0', 'AVG(Spirituality)', results.iloc[0]['AVG(Spirituality)']],
        ['g0', 'AVG(Teamwork)', results.iloc[0]['AVG(Teamwork)']],
        ['g0', 'AVG(Zest)', results.iloc[0]['AVG(Zest)']],

        # Second row.
        ['g1', 'AVG(Openness)', results.iloc[1]['AVG(Openness)']],
        ['g1', 'AVG(Restraint)', results.iloc[1]['AVG(Restraint)']],
        ['g1', 'AVG(Transcendence)', results.iloc[1]['AVG(Transcendence)']],
        ['g1', 'AVG(Interpersonal)', results.iloc[1]['AVG(Interpersonal)']],
        ['g1', 'AVG(GHQ_12)', results.iloc[1]['AVG(GHQ_12)']],
        ['g1', 'AVG(SEC)', results.iloc[0]['AVG(SEC)']],
        ['g1', 'AVG(Age)', results.iloc[1]['AVG(Age)']],
        ['g1', 'AVG(Work)', results.iloc[1]['AVG(Work)']],
        ['g1', 'AVG(Day)', results.iloc[1]['AVG(Day)']],
        ['g1', 'AVG(Sons)', results.iloc[1]['AVG(Sons)']],
        ['g1', 'AVG(Appreciation_of_beauty)', results.iloc[1]['AVG(Appreciation_of_beauty)']],
        ['g1', 'AVG(Bravery)', results.iloc[1]['AVG(Bravery)']],
        ['g1', 'AVG(Creativity)', results.iloc[1]['AVG(Creativity)']],
        ['g1', 'AVG(Curiosity)', results.iloc[1]['AVG(Curiosity)']],
        ['g1', 'AVG(Fairness)', results.iloc[1]['AVG(Fairness)']],
        ['g1', 'AVG(Gratitude)', results.iloc[1]['AVG(Gratitude)']],
        ['g1', 'AVG(Honesty)', results.iloc[1]['AVG(Honesty)']],
        ['g1', 'AVG(Hope)', results.iloc[1]['AVG(Hope)']],
        ['g1', 'AVG(Humilty)', results.iloc[1]['AVG(Humilty)']],
        ['g1', 'AVG(Humor)', results.iloc[1]['AVG(Humor)']],
        ['g1', 'AVG(Judgment)', results.iloc[1]['AVG(Judgment)']],
        ['g1', 'AVG(Kindness)', results.iloc[1]['AVG(Kindness)']],
        ['g1', 'AVG(Leadership)', results.iloc[1]['AVG(Leadership)']],
        ['g1', 'AVG(Love)', results.iloc[1]['AVG(Love)']],
        ['g1', 'AVG(Love_of_learning)', results.iloc[1]['AVG(Love_of_learning)']],
        ['g1', 'AVG(Perseverance)', results.iloc[1]['AVG(Perseverance)']],
        ['g1', 'AVG(Perspective)', results.iloc[1]['AVG(Perspective)']],
        ['g1', 'AVG(Prudence)', results.iloc[1]['AVG(Prudence)']],
        ['g1', 'AVG(Self_regulation)', results.iloc[1]['AVG(Self_regulation)']],
        ['g1', 'AVG(Social_intelligence)', results.iloc[1]['AVG(Social_intelligence)']],
        ['g1', 'AVG(Spirituality)', results.iloc[1]['AVG(Spirituality)']],
        ['g1', 'AVG(Teamwork)', results.iloc[1]['AVG(Teamwork)']],
        ['g1', 'AVG(Zest)', results.iloc[1]['AVG(Zest)']],

        # Third row.
        ['g2', 'AVG(Openness)', results.iloc[2]['AVG(Openness)']],
        ['g2', 'AVG(Restraint)', results.iloc[2]['AVG(Restraint)']],
        ['g2', 'AVG(Transcendence)', results.iloc[2]['AVG(Transcendence)']],
        ['g2', 'AVG(Interpersonal)', results.iloc[2]['AVG(Interpersonal)']],
        ['g2', 'AVG(GHQ_12)', results.iloc[2]['AVG(GHQ_12)']],
        ['g2', 'AVG(SEC)', results.iloc[0]['AVG(SEC)']],
        ['g2', 'AVG(Age)', results.iloc[2]['AVG(Age)']],
        ['g2', 'AVG(Work)', results.iloc[2]['AVG(Work)']],
        ['g2', 'AVG(Day)', results.iloc[2]['AVG(Day)']],
        ['g2', 'AVG(Sons)', results.iloc[2]['AVG(Sons)']],
        ['g2', 'AVG(Appreciation_of_beauty)', results.iloc[2]['AVG(Appreciation_of_beauty)']],
        ['g2', 'AVG(Bravery)', results.iloc[2]['AVG(Bravery)']],
        ['g2', 'AVG(Creativity)', results.iloc[2]['AVG(Creativity)']],
        ['g2', 'AVG(Curiosity)', results.iloc[2]['AVG(Curiosity)']],
        ['g2', 'AVG(Fairness)', results.iloc[2]['AVG(Fairness)']],
        ['g2', 'AVG(Gratitude)', results.iloc[2]['AVG(Gratitude)']],
        ['g2', 'AVG(Honesty)', results.iloc[2]['AVG(Honesty)']],
        ['g2', 'AVG(Hope)', results.iloc[2]['AVG(Hope)']],
        ['g2', 'AVG(Humilty)', results.iloc[2]['AVG(Humilty)']],
        ['g2', 'AVG(Humor)', results.iloc[2]['AVG(Humor)']],
        ['g2', 'AVG(Judgment)', results.iloc[2]['AVG(Judgment)']],
        ['g2', 'AVG(Kindness)', results.iloc[2]['AVG(Kindness)']],
        ['g2', 'AVG(Leadership)', results.iloc[2]['AVG(Leadership)']],
        ['g2', 'AVG(Love)', results.iloc[2]['AVG(Love)']],
        ['g2', 'AVG(Love_of_learning)', results.iloc[2]['AVG(Love_of_learning)']],
        ['g2', 'AVG(Perseverance)', results.iloc[2]['AVG(Perseverance)']],
        ['g2', 'AVG(Perspective)', results.iloc[2]['AVG(Perspective)']],
        ['g2', 'AVG(Prudence)', results.iloc[2]['AVG(Prudence)']],
        ['g2', 'AVG(Self_regulation)', results.iloc[2]['AVG(Self_regulation)']],
        ['g2', 'AVG(Social_intelligence)', results.iloc[2]['AVG(Social_intelligence)']],
        ['g2', 'AVG(Spirituality)', results.iloc[2]['AVG(Spirituality)']],
        ['g2', 'AVG(Teamwork)', results.iloc[2]['AVG(Teamwork)']],
        ['g2', 'AVG(Zest)', results.iloc[2]['AVG(Zest)']]],
        columns=['group', 'column', 'val'])
    df.pivot("group", "column", "val").plot(kind='bar', legend=None)
    plt.title("Group bar graph of all categories under different DASS21 score groups")
    plt.ylabel("Average score for each category")
    plt.show()


def makePrediction(inputDf, PATH):
    """
    Make prediction base on the dataset.
    :param inputDf: dataframe
    :param PATH: str
    """
    const = 5.9308
    Openness = 0.1405
    GHQ_12 = 0.7211
    SEC = -0.6325
    Work = 0.6157
    Hope = -0.7198
    AgeBin = 2.6940
    newDf = pd.DataFrame()

    for i in range(0, len(inputDf)):
        if inputDf.iloc[i]['Age'] >= 18 or inputDf.iloc[i]['Age'] < 29:
            is_age = 1
        else:
            is_age = 0

        prediction = const + Openness * inputDf.iloc[i]['Openness'] + GHQ_12 * inputDf.iloc[i]['GHQ_12'] \
                     + SEC * inputDf.iloc[i]['SEC'] + Hope * inputDf.iloc[i]['Hope'] \
                     + Work * inputDf.iloc[i]['Work'] + AgeBin * is_age

        newDf = newDf.append({"DASS_21": prediction}, ignore_index=True)
    print(newDf)
    newDf.to_csv(PATH + "DASS21_output.csv")


def main():
    """
    Drives the program.
    """
    # ============================ Change the code below to import data =========================
    # change your path here
    PATH = "/Users/xindilu/Desktop/COMP 3948 Modelling 📊📈/A1/"

    print("\n\nRead the input file and output predictions into a csv file.\n")
    mysterInputDf = pd.read_csv(PATH + "DASS21_mystery_input.csv")
    makePrediction(mysterInputDf, PATH)
    print("\nResult file is exported")

    # ============================= visualize the multi distribution graphs ======================
    # print("\n\nVisualize the Category distribution graphs for all DASS_21 groups")
    # model_visual(PATH)

    # ================== visualize the Group bar graph of all categories under different DASS21 score groups
    # print("\n\nVisualize the Group bar graph of all categories under different DASS21 score groups.")
    # print_DASS21_score_in_group(PATH)


if __name__ == '__main__':
    main()
