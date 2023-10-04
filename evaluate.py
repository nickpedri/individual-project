from math import sqrt

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

"""
Cheat sheet for stats tests:

Chi-Square test  -  Used to test TWO CATEGORICAL types of data.
Pearson's R  -  Used to test TWO CONTINUOUS types of data.
Spearman's R  -  Used to test TWO CONTINUOUS types of data.
T-Test  -  Used to test ONE CONTINUOUS AND ONE CATEGORICAL type of data.
Mann-Whitney  -  Used to test ONE CONTINUOUS AND ONE CATEGORICAL type of data.

Parametric tests relies on statistical distributions in data.
Non-Parametric tests do not rely on any kind of distribution.
"""


def eval_p(p_value):  # create a function that checks p-value for a significant result.
    """This function simply takes in a p-value checks if it is greater than or less than alpha."""
    alpha = 0.05  # set confidence interval
    if p_value < alpha:
        print(f'There is a signifcant result. P-value was {round(p_value,2)}.')
    else:
        print(f'There is no signifcant result. P-value was {round(p_value,2)}.')


def eval_pearson_r(r_value, p_val):
    """This function will evaluate Pearson's R."""
    eval_p(p_val)
    if r_value >= 0.90:
        print(f'There is a very high positive correlation. R-value was {round(r_value,3)}.')
    elif r_value >= 0.70:
        print(f'There is a high positive correlation. R-value was {round(r_value,3)}.')
    elif r_value >= 0.40:
        print(f'There is a moderate positive correlation. R-value was {round(r_value,3)}.')
    elif r_value >= 0.20:
        print(f'There is a low positive correlation. R-value was {round(r_value,3)}.')
    elif r_value > 0.00:
        print(f'There is a very slight positive correlation. R-value was {round(r_value,3)}.')
    elif r_value == 0.00:
        print(f'There is no correlation. R-value was {round(r_value,3)}.')
    elif r_value >= -0.20:
        print(f'There is a very slight negative correlation. R-value was {round(r_value,3)}.')
    elif r_value >= -0.40:
        print(f'There is a low negative correlation. R-value was {round(r_value,3)}.')
    elif r_value >= -0.70:
        print(f'There is a moderate negative correlation. R-value was {round(r_value,3)}.')
    elif r_value >= -0.90:
        print(f'There is a high negative correlation. R-value was {round(r_value,3)}.')
    else:
        print(f'There is a very high negative correlation. R-value was {round(r_value, 2)}.')


def check_ttest(t, p, tails=1):
    """This function will accept the t, and p-value from a T-test and analyze it. It will accordingly divide the p-value
    by 2 if this is a one-tailed test."""
    alpha = 0.05
    if tails == 1:
        if p / 2 > alpha:
            print(f'There is no signifcant result. P-value was {round(p,2)}.')
        else:
            print(f'There is a signifcant result. P-value was {round(p, 2)}.')
    if tails == 2:
        if p > alpha:
            print(f'There is no signifcant result. P-value was {round(p,2)}.')
        else:
            print(f'There is a signifcant result. P-value was {round(p, 2)}.')
    if t > 0:
        print(f'T-value was greater than 0. With a value of {round(t,2)}.')
    else:
        print(f'T-value was less than 0. With a value of {round(t,2)}.')


def baseline(data, method='both'):
    """This function will create a baseline. It accepts a dataframe and then returns one or two baseline models."""
    df = pd.DataFrame(data)  # Creates a copy of dataframe

    if method == 'mean':  # Creates a baseline model using mean()
        df['baseline'] = data.mean()
        return df
    elif method == 'median':  # Creates a baseline model using median()
        df['baseline'] = data.median()
        return df
    elif method == 'both':  # Creates a baseline model using both
        df['base_median'] = data.median()
        df['base_mean'] = data.mean()
        return df  # Returns dataframe with the baseline models


def baseline_classification(dataframe, data='', target_value='', show_results=True):
    results = pd.DataFrame(dataframe[data])
    results['baseline'] = dataframe[data].mode()[0]
    df = dataframe.copy()
    df['baseline'] = df[data].mode()[0]  # Creates a baseline prediction from the mode of the data
    b_acc = (df[data] == df['baseline']).mean()
    sub_rec = df[df[data] == target_value]  # Subset of all positive cases for recall
    b_rec = (sub_rec[data] == sub_rec['baseline']).mean()
    sub_bas_pre = df[df['baseline'] == target_value]
    if sub_bas_pre.empty:
        bas_pre = 0.0
    else:
        bas_pre = (sub_bas_pre[data] == sub_bas_pre['baseline']).mean()
    if show_results:
        print(f'Baseline accuracy is: {round(b_acc * 100, 2)}%.')
        print(f'Baseline recall is: {round(b_rec * 100, 2)}%.')
        print(f'Baseline precision is: {round(bas_pre * 100, 2)}%.')
        print()
    return results


def eval_model(actual, model):
    """This function will accept two series of the model and actual data and calculate the metrics for the model."""
    residuals = model - actual  # Calculate residuals
    SSE = (residuals ** 2).sum()
    MSE = SSE / len(actual)
    RMSE = sqrt(MSE)
    return SSE, MSE, RMSE  # Returns the calculated metrics


def train_model(model, X_train, y_train, X_val, y_val, X_test=pd.DataFrame, y_test=pd.DataFrame, test=False):
    """This function accepts a model object, and the x and y train and validate dataframes. It will fit, predict, and
    evaluate the models on train and validate."""
    model.fit(X_train, y_train)  # Fits the model to the train data
    print(model.score(X_train, y_train))
    print(model.score(X_val, y_val))
    if test is True:
        print(model.score(X_test, y_test))
    # train_preds = model.predict(X_train)  # Create predictions for train
    # val_preds = model.predict(X_val)  # Creates predictions for validate


def train_model_gen2(model, X_train, y_train, X_val, y_val):
    """This function accepts a model object, and the x and y train and validate dataframes. It will fit, predict, and
    evaluate the models on train and validate."""
    model.fit(X_train, y_train)  # Fits the model to the train data
    train_preds = model.predict(X_train).round()  # Create predictions for train
    skip, skip2, train_rmse = eval_model(y_train, train_preds)  # Caculate RMSE for model on train
    val_preds = model.predict(X_val).round()  # Creates predictions for validate
    skip3, skip4, val_rmse = eval_model(y_val, val_preds)  # Caculate RMSE for model on validate
    print(f'The train RMSE is {train_rmse}.')
    print(f'The validate RMSE is {val_rmse}.')


def distributions(df):
    for col in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col])
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()
