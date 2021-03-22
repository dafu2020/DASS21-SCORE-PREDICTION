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
Modelling code for the best model which runs from start to finish without error. 
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


def acquire_data_description(dataset):
    """
    Display the general statistic information of the dataset.
    Including:
        1. data types
        2. Stat summaries for numeric columns
        3. Stat summaries for non-numeric columns
    :param dataset: a dataset
    """
    print(dataset.head())

    # print title with space before.
    print("\n===DATA TYPES===")
    print(dataset.dtypes)

    # Show statistical summaries for numeric columns.
    print("\n===STATISTIC SUMMARIES for NUMERIC COLUMNS===")
    print(dataset.describe().transpose())

    # Show summaries for objects like dates and strings.
    print("\n===STATISTIC SUMMARIES for NON NUMERIC COLUMNS===")
    print(dataset.describe(include=['object']).transpose())


def generate_heatmap(dataset, title):
    """
    Generate a heatmap to display the correlation of the predictors.
    :param dataset: a dataset
    :param title: str
    :return: a heatmap
    """
    corr = dataset.corr()
    sns.heatmap(corr, xticklabels=corr.columns,
                yticklabels=corr.columns)

    plt.title(title)
    plt.show()


def model_raw(PATH):
    """
    This model generate the overview of the raw data.
    """
    # load data
    df = import_data(PATH)

    # display dataset stat summaries of the dataset
    acquire_data_description(df)

    # Store x and y values.
    X = df[['Openness', 'Restraint', 'Transcendence', 'Interpersonal', 'GHQ_12', 'SEC',
            'Age', 'Work', 'Day', 'Sons', 'Appreciation_of_beauty', 'Bravery', 'Creativity',
            'Curiosity', 'Fairness', 'Forgiveness', 'Gratitude', 'Honesty', 'Hope', 'Humilty',
            'Humor', 'Judgment', 'Kindness', 'Leadership', 'Love', 'Love_of_learning', 'Perseverance',
            'Perspective', 'Prudence', 'Self_regulation', 'Social_intelligence', 'Spirituality',  # 32
            'Teamwork', 'Zest']]

    X = sm.add_constant(X)
    target = df['DASS_21']

    # Create training set with 80% of data and test set with 20% of data.
    X_train, X_test, y_train, y_test = train_test_split(X, target, train_size=0.8, random_state=0)

    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model

    print(model.summary())

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    rsquare = model.rsquared

    return rmse, rsquare


def model_one(PATH):
    """
    This model do not contain binned and dummy variables.
    """
    # load data
    df = import_data(PATH)

    # display dataset stat summaries of the dataset
    # acquire_data_description(df)

    # print(df.head())

    # Store x and y values.
    X = df[['Openness', 'GHQ_12', 'SEC', 'Age',
            'Work', 'Appreciation_of_beauty', 'Forgiveness',
            'Hope']]

    X = sm.add_constant(X)
    target = df['DASS_21']

    # Create training set with 80% of data and test set with 20% of data.
    X_train, X_test, y_train, y_test = train_test_split(X, target, train_size=0.8, random_state=0)

    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model

    print(model.summary())
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    selected_predictors = df[['DASS_21', 'Openness', 'GHQ_12', 'SEC', 'Age',
                              'Work', 'Appreciation_of_beauty', 'Forgiveness',
                              'Hope']]
    generate_heatmap(selected_predictors, "model one")

    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    rsquare = model.rsquared

    return rmse, rsquare


def model_two(PATH):
    """
    This model contain binned and dummy variables.
    """
    # load data
    df = import_data(PATH)

    # display dataset stat summaries of the dataset
    acquire_data_description(df)

    df['AgeBin'] = pd.cut(x=df['Age'], bins=[18, 29, 40, 51, 62, 73, 84])
    df['RestraintBin'] = pd.cut(x=df['Restraint'], bins=[20, 35, 50, 65, 80])
    df['WorkBin'] = pd.cut(x=df['Work'], bins=[0, 2, 4, 7])
    df['Appreciation_of_beautyBin'] = pd.cut(x=df['Appreciation_of_beauty'], bins=[10, 20, 30])
    df['SpiritualityBin'] = pd.cut(x=df['Spirituality'], bins=[0, 10, 20, 30])
    tempDf = df[['AgeBin', 'Gender', 'Student', 'RestraintBin', 'WorkBin', 'Appreciation_of_beautyBin',
                 'SpiritualityBin']]
    df_dummy = pd.get_dummies(tempDf, columns=['AgeBin', 'Gender', 'Student', 'RestraintBin', 'WorkBin',
                                               'Appreciation_of_beautyBin', 'SpiritualityBin'])
    df = pd.concat(([df, df_dummy]), axis=1)
    # print(df.head())

    # significant predictors
    X = df[['Openness', 'GHQ_12', 'SEC',
            'Forgiveness', 'Hope',
            'Prudence', 'Zest', 'Gender_Female', 'Gender_Male', 'Student_Other',
            'Student_Student',
            'Appreciation_of_beautyBin_(10, 20]', 'Appreciation_of_beautyBin_(20, 30]']]

    X = sm.add_constant(X)
    target = df['DASS_21']

    # Create training set with 80% of data and test set with 20% of data.
    X_train, X_test, y_train, y_test = train_test_split(X, target, train_size=0.8)

    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model

    print(model.summary())
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    selected_predictors = df[['DASS_21', 'Openness', 'GHQ_12', 'SEC',
                              'Forgiveness', 'Hope',
                              'Prudence', 'Zest', 'Gender_Female', 'Gender_Male', 'Student_Other',
                              'Student_Student',
                              'Appreciation_of_beautyBin_(10, 20]', 'Appreciation_of_beautyBin_(20, 30]']]
    generate_heatmap(selected_predictors, "model two")

    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    rsquare = model.rsquared

    return rmse, rsquare


def viewAndGetOutliersByPercentile(df, colName, lowerP, upperP, plt):
    """
    Detect the outlier that is not within the 95% of data sets.
    :param df: dataframe
    :param colName: str
    :param lowerP: float
    :param upperP: float
    :param plt: plt
    :return: float
    """
    # Show basic statistics.
    dfSub = df[[colName]]
    # print("*** Statistics for " + colName)
    # print(dfSub.describe())

    # Show boxplot.
    dfSub.boxplot(column=[colName])
    plt.title(colName)
    # plt.show()

    # Get upper and lower perctiles and filter with them.
    up = df[colName].quantile(upperP)
    lp = df[colName].quantile(lowerP)
    outlierDf = df[(df[colName] < lp) | (df[colName] > up)]

    # Show filtered and sorted DataFrame with outliers.
    dfSorted = outlierDf.sort_values([colName], ascending=[True])
    # print("\nDataFrame rows containing outliers for " + colName + ":")
    # print(dfSorted)

    return lp, up  # return lower and upper percentiles


def model_three(PATH):
    """
    This model contain outlier treatment.
    """
    df = import_data(PATH)

    # display dataset stat summaries of the dataset
    # acquire_data_description(df)

    df['AgeBin'] = pd.cut(x=df['Age'], bins=[18, 29, 40, 51, 62, 73, 84])
    df['RestraintBin'] = pd.cut(x=df['Restraint'], bins=[20, 35, 50, 65, 80])
    df['Appreciation_of_beautyBin'] = pd.cut(x=df['Appreciation_of_beauty'], bins=[10, 20, 30])
    tempDf = df[['AgeBin', 'Gender', 'Student', 'RestraintBin', 'Appreciation_of_beautyBin']]
    df_dummy = pd.get_dummies(tempDf, columns=['AgeBin', 'Gender', 'Student', 'RestraintBin',
                                               'Appreciation_of_beautyBin'])
    df = pd.concat(([df, df_dummy]), axis=1)
    # print(df.head())

    # detect outliers in the data
    LOWER_PERCENTILE = 0.00025
    UPPER_PERCENTILE = 0.99925

    # Openness, Transcendence, Interpersonal, and Age
    Openness_lp, Openness_up = viewAndGetOutliersByPercentile(df, 'Openness', LOWER_PERCENTILE, UPPER_PERCENTILE, plt)

    Transcendence_lp, Transcendence_up = viewAndGetOutliersByPercentile(df, 'Transcendence', LOWER_PERCENTILE,
                                                                        UPPER_PERCENTILE, plt)

    Interpersonal_lp, Interpersonal_up = viewAndGetOutliersByPercentile(df, 'Interpersonal', LOWER_PERCENTILE,
                                                                        UPPER_PERCENTILE, plt)

    Age_lp, Age_up = viewAndGetOutliersByPercentile(df, 'Age', LOWER_PERCENTILE, UPPER_PERCENTILE, plt)

    # filter the data set that does not contain outliers
    df_filtered = df[(df["Openness"] > Openness_lp) & (df["Openness"] < Openness_up) &
                     (df["Transcendence"] > Transcendence_lp) & (df["Transcendence"] < Transcendence_up) &
                     (df["Interpersonal"] > Interpersonal_lp) & (df["Interpersonal"] < Interpersonal_up) &
                     (df["Age"] > Age_lp) & (df["Age"] < Age_up)]

    # print(df_filtered.describe())

    # significant predictors
    X = df_filtered[
        ['Openness', 'GHQ_12', 'SEC', 'Work', 'Hope', 'AgeBin_(18, 29]']]

    X = sm.add_constant(X)
    target = df_filtered['DASS_21'].values

    # Create training set with 80% of data and test set with 20% of data.
    X_train, X_test, y_train, y_test = train_test_split(X, target, train_size=0.8)

    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model

    print(model.summary())
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    predictors_selected = df_filtered[['DASS_21', 'Openness', 'GHQ_12', 'SEC', 'Work', 'Hope', 'AgeBin_(18, 29]']]
    generate_heatmap(predictors_selected, "Model 3")

    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    rsquare = model.rsquared

    BINS = 8
    TITLE = "DASS21 Score"
    drawValidationPlots(TITLE, BINS, y_test, predictions)

    return rmse, rsquare


def model_four(PATH):
    """
    This model contain binned, dummy variables, outlier handling and transformed variables.
    """
    # load data
    df = import_data(PATH)

    # display dataset stat summaries of the dataset
    # acquire_data_description(df)

    tempDf = df[['Gender', 'Student']]
    df_dummy = pd.get_dummies(tempDf, columns=['Gender', 'Student'])
    df = pd.concat(([df, df_dummy]), axis=1)
    # print(df.head())

    # detect outliers in the data
    LOWER_PERCENTILE = 0.00025
    UPPER_PERCENTILE = 0.99925

    # Openness, Transcendence, Interpersonal, and Age
    Openness_lp, Openness_up = viewAndGetOutliersByPercentile(df, 'Openness', LOWER_PERCENTILE, UPPER_PERCENTILE, plt)

    Transcendence_lp, Transcendence_up = viewAndGetOutliersByPercentile(df, 'Transcendence', LOWER_PERCENTILE,
                                                                        UPPER_PERCENTILE, plt)

    Interpersonal_lp, Interpersonal_up = viewAndGetOutliersByPercentile(df, 'Interpersonal', LOWER_PERCENTILE,
                                                                        UPPER_PERCENTILE, plt)

    Age_lp, Age_up = viewAndGetOutliersByPercentile(df, 'Age', LOWER_PERCENTILE, UPPER_PERCENTILE, plt)

    # filter the data set that does not contain outliers
    df_filtered = df[(df["Openness"] > Openness_lp) & (df["Openness"] < Openness_up) &
                     (df["Transcendence"] > Transcendence_lp) & (df["Transcendence"] < Transcendence_up) &
                     (df["Interpersonal"] > Interpersonal_lp) & (df["Interpersonal"] < Interpersonal_up) &
                     (df["Age"] > Age_lp) & (df["Age"] < Age_up)]

    # Store x and y values.
    X = df_filtered[{'Transcendence', 'GHQ_12', 'SEC', 'Age', 'Work', 'Sons', 'Bravery', 'Creativity', 'Curiosity',
                     'Fairness', 'Forgiveness', 'Gratitude', 'Hope', 'Humilty', 'Humor',
                     'Kindness', 'Leadership', 'Love', 'Love_of_learning', 'Perseverance', 'Perspective',
                     'Prudence', 'Appreciation_of_beauty',
                     'Self_regulation', 'Social_intelligence', 'Spirituality', 'Teamwork', 'Zest',
                     'Gender_Female',
                     'Gender_Male', 'Student_Other', 'Student_Student'}]

    # new X set containing transformed data
    X_transformed = df_filtered[{'GHQ_12', 'SEC', 'Hope', 'Gender_Female', 'Gender_Male', 'Student_Other',
                                 'Student_Student'}]

    X_TransformedAge = stats.boxcox(X['Age'])
    X_transformed['transformedAge'] = X_TransformedAge[0]

    X_TransformedBeauty = stats.boxcox(X['Appreciation_of_beauty'])
    X_transformed['transformedAppreciation_of_beauty'] = X_TransformedBeauty[0]

    # X_TransformedJudgment = stats.boxcox(X['Judgment'])
    # X_transformed['transformedJudgment'] = X_TransformedJudgment[0]
    #
    # X_TransformedHonesty = stats.boxcox(X['Honesty'])
    # X_transformed['transformedHonesty'] = X_TransformedHonesty[0]
    # X_TransformedDay = stats.boxcox(X['Day'])
    # X_transformed['transformedDay'] = X_TransformedDay[0]

    X_transformed = sm.add_constant(X_transformed)
    target = df_filtered['DASS_21']

    # Create training set with 80% of data and test set with 20% of data.
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, target, train_size=0.8)

    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model

    # ======================================== COMMENT OUT NEXT TWO LINES FOR MODEL SUMMARY =========================
    print(model.summary())
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    # predictors_selected = X_transformed[
    #     ['GHQ_12', 'SEC', 'Hope', 'Gender_Female', 'Gender_Male', 'Student_Other',
    #      'Student_Student']]
    #
    # # ======================================== COMMENT OUT NEXT LINE FOR MODEL CORRELATION HEATMAP ===================
    # generate_heatmap(predictors_selected, "Model 4 Correlation Map")

    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    rsquare = model.rsquared

    return rmse, rsquare


def model_five(PATH):
    """
    This model contain binned, dummy variables and transformed variables.
    """
    # load data
    df = import_data(PATH)

    # display dataset stat summaries of the dataset
    # acquire_data_description(df)

    tempDf = df[['Gender', 'Student']]
    df_dummy = pd.get_dummies(tempDf, columns=['Gender', 'Student'])
    df = pd.concat(([df, df_dummy]), axis=1)
    # print(df.head())

    # Store x and y values.
    X = df[['Openness', 'Restraint', 'Transcendence', 'Interpersonal', 'GHQ_12', 'Age', 'SEC',
            'Work', 'Day', 'Sons', 'Appreciation_of_beauty', 'Bravery', 'Creativity', 'Curiosity',
            'Fairness', 'Forgiveness', 'Gratitude', 'Honesty', 'Hope', 'Humilty', 'Humor', 'Judgment',
            'Kindness', 'Leadership', 'Love', 'Love_of_learning', 'Perseverance', 'Perspective', 'Prudence',
            'Self_regulation', 'Social_intelligence', 'Spirituality', 'Teamwork', 'Zest', 'Gender_Female',
            'Gender_Male', 'Student_Other', 'Student_Student']]

    # new X set containing transformed data
    X_transformed = df[['Openness', 'Transcendence', 'GHQ_12', 'SEC',
                        'Hope', 'Gender_Female',
                        'Gender_Male', 'Student_Other', 'Student_Student']]

    X_TransformedAge = stats.boxcox(X['Age'])
    X_transformed['transformedAge'] = X_TransformedAge[0]

    X_TransformedBeauty = stats.boxcox(X['Appreciation_of_beauty'])
    X_transformed['transformedAppreciation_of_beauty'] = X_TransformedBeauty[0]

    # X_TransformedDay = stats.boxcox(X['Day'])
    # X_transformed['transformedDay'] = X_TransformedDay[0]

    # X_TransformedJudgment = stats.boxcox(X['Judgment'])
    # X_transformed['transformedJudgment'] = X_TransformedJudgment[0]

    # X_TransformedHonesty = stats.boxcox(X['Honesty'])
    # X_transformed['transformedHonesty'] = X_TransformedHonesty[0]

    X_transformed = sm.add_constant(X_transformed)
    target = df['DASS_21']

    # Create training set with 80% of data and test set with 20% of data.
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, target, train_size=0.8)

    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model

    # ========================================COMMENT OUT NEXT TWO LINES FOR MODEL SUMMARY!!=========================
    print(model.summary())
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
    #
    # # generate heatmap
    # predictors_selected = X_transformed[['Openness', 'Transcendence', 'GHQ_12', 'SEC',
    #                                      'Hope', 'Gender_Female',
    #                                      'Gender_Male', 'Student_Other', 'Student_Student',
    #                                      'transformedAppreciation_of_beauty',
    #                                      'transformedAge']]
    # title = "Model 5 Correlation Map"
    # generate_heatmap(predictors_selected, title)

    rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    rsquare = model.rsquared

    return rmse, rsquare


def performLinearRegression_iteratively_model_5(PATH, number_run):
    """
    Perform Linear Regression iteratively for model 5.
    :param PATH: str
    :param number_run: int
    """
    result_list = []
    avg_rmse = 0
    avg_rsquare = 0
    count = 0
    while count < number_run:
        rmse, rsquare = model_five(PATH)
        count += 1
        result = f"RMSE = {rmse}, rsquare = {rsquare}"
        result_list.append(result)
        avg_rmse += rmse
        avg_rsquare += rsquare

    print(f"\nModel 5 with {number_run} runs:")
    for result in result_list:
        print(result)
    print(f"Avg RMSE  = {avg_rmse / number_run}, Avg Rsquare = {avg_rsquare / number_run}\n")


def performLinearRegression_iteratively_model_4(PATH, number_run):
    """
    Perform Linear Regression iteratively for model 4.
    :param PATH: str
    :param number_run: int
    """
    result_list = []
    avg_rmse = 0
    avg_rsquare = 0
    count = 0
    while count < number_run:
        rmse, rsquare = model_four(PATH)
        count += 1
        result = f"RMSE = {rmse}, rsquare = {rsquare}"
        result_list.append(result)
        avg_rmse += rmse
        avg_rsquare += rsquare

    print(f"\nModel 4 with {number_run} runs:")
    for result in result_list:
        print(result)
    print(f"Avg RMSE  = {avg_rmse / number_run}, Avg Rsquare = {avg_rsquare / number_run}\n")


def performLinearRegression_iteratively_model_3(PATH, number_run):
    """
    Perform Linear Regression iteratively for model 3.
    :param PATH: str
    :param number_run: int
    """
    result_list = []
    avg_rmse = 0
    avg_rsquare = 0
    count = 0
    while count < number_run:
        rmse, rsquare = model_three(PATH)
        count += 1
        result = f"RMSE = {rmse}, rsquare = {rsquare}"
        result_list.append(result)
        avg_rmse += rmse
        avg_rsquare += rsquare

    print(f"\nModel 3 with {number_run} runs:")
    for result in result_list:
        print(result)
    print(f"Avg RMSE  = {avg_rmse / number_run}, Avg Rsquare = {avg_rsquare / number_run}\n")


def plotPredictionVsActual(plt, title, y_test, predictions):
    plt.scatter(y_test, predictions)
    plt.legend()
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title('Predicted (Y) vs. Actual (X): ' + title)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')


def plotResidualsVsActual(plt, title, y_test, predictions):
    residuals = y_test - predictions
    plt.scatter(y_test, residuals, label='Residuals vs Actual')
    plt.xlabel("Actual")
    plt.ylabel("Residual")
    plt.title('Error Residuals (Y) vs. Actual (X): ' + title)
    plt.plot([y_test.min(), y_test.max()], [0, 0], 'k--')


def plotResidualHistogram(plt, title, y_test, predictions, bins):
    residuals = y_test - predictions
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.hist(residuals, label='Residuals vs Actual', bins=bins)
    plt.title('Error Residual Frequency: ' + title)
    plt.plot(legend = None)


def drawValidationPlots(title, bins, y_test, predictions):
    # Define number of rows and columns for graph display.
    plt.subplots(nrows=1, ncols=3, figsize=(12, 5))

    plt.subplot(1, 3, 1)  # Specfy total rows, columns and image #
    plotPredictionVsActual(plt, title, y_test, predictions)

    plt.subplot(1, 3, 2)  # Specfy total rows, columns and image #
    plotResidualsVsActual(plt, title, y_test, predictions)

    plt.subplot(1, 3, 3)  # Specfy total rows, columns and image #
    plotResidualHistogram(plt, title, y_test, predictions, bins)
    plt.show()


def main():
    """
    Drives the program.
    """

    # ============================ Change the code below to import data =========================
    # change your path here
    PATH = "/Users/xindilu/Desktop/COMP 3948 Modelling 📊📈/A1/"

    # ======= Display the OLS linear regression summary of the selected top model to make predictions ========
    print("\n\nDisplay the OLS linear regression summary of the selected top model to make predictions")
    model_three(PATH)

    # ============================= performing linear regression iteratively ====================
    # print("\nPerforming linear regression iteratively for Model 3:")
    # performLinearRegression_iteratively_model_3(PATH, 10)

    # ================== Print other candidate models =============================================
    # print("\n\n========== Display model 1")
    # model_one(PATH)
    # print("\n\n========== Display model 2")
    # model_two(PATH)
    # print("\n\n========== Display model 4")
    # model_four(PATH)
    # print("\n\n========== Display model 5")
    # model_five(PATH)


if __name__ == '__main__':
    main()
